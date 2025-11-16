from PIL import Image
import os
from datetime import datetime
import requests
import io
import time
import logging
import signal
import sys
import json
from collections import deque
from picamera2 import Picamera2
import schedule

# Configuration
CLOUD_FUNCTION_URL = "https://europe-west3-pressure-watcher-ea41a.cloudfunctions.net/process-image"
CAPTURE_INTERVAL = 5  # minutes
RETRY_INTERVAL = 30  # seconds
MAX_RETRIES = 5
BACKOFF_MULTIPLIER = 2

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

# These will be loaded from config file
DEVICE_ID = None
GAUGE_MIN = None
GAUGE_MAX = None
GAUGE_UNIT = None
DEBUG_IMAGE_PATH = None
LOG_FILE = None


def load_config():
    """Load configuration from JSON file."""
    global DEVICE_ID, GAUGE_MIN, GAUGE_MAX, GAUGE_UNIT

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)

        DEVICE_ID = config['device_id']
        GAUGE_MIN = config['gauge']['min_value']
        GAUGE_MAX = config['gauge']['max_value']
        GAUGE_UNIT = config['gauge']['unit']
        DEBUG_IMAGE_PATH = config['debug_image_path']
        LOG_FILE = config['log_file']

        logger.info(f"Configuration loaded from {CONFIG_FILE}")
        logger.info(f"Device: {DEVICE_ID}, Range: {GAUGE_MIN}-{GAUGE_MAX} {GAUGE_UNIT}")
        logger.info(f"Log file output to: {LOG_FILE}.")
        logger.info(f"Debug image output to: {DEBUG_IMAGE_PATH}.")

    except FileNotFoundError:
        logger.error(f"Config file not found: {CONFIG_FILE}")
        logger.error("Please create config.json with device_id and gauge settings")
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Missing required config key: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Failed upload queue: stores (image, timestamp, retry_count)
failed_uploads = deque()

# Global camera instance
camera = None


def capture_image():
    """Capture image from Pi camera and return as PIL Image."""
    global camera

    try:
        if camera is None:
            camera = Picamera2()
            camera.configure(camera.create_still_configuration())
            camera.start()
            time.sleep(2)  # Camera warm-up
            logger.info("Camera initialized")

        # Capture image as numpy array and convert to PIL
        image_array = camera.capture_array()
        image = Image.fromarray(image_array)

        # Save debug copy
        image.save(DEBUG_IMAGE_PATH)
        logger.info(f"Image captured and saved to {DEBUG_IMAGE_PATH}")

        return image

    except Exception as e:
        logger.error(f"Camera capture failed: {e}")
        return None


def upload_image(image, timestamp):
    """Upload image to cloud function. Returns True if successful."""
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"

        files = {'image': (filename, img_byte_arr, 'image/jpeg')}
        data = {
            'image': filename,
            'timestamp': str(timestamp),
            'device_id': DEVICE_ID,
            'min_value': GAUGE_MIN,
            'max_value': GAUGE_MAX,
            'unit': GAUGE_UNIT
        }

        response = requests.post(CLOUD_FUNCTION_URL, files=files, data=data, timeout=30)

        if response.status_code == 200:
            logger.info(f"Upload successful: {filename}")
            logger.debug(f"Response: {response.text}")
            return True
        else:
            logger.warning(f"Upload failed with status {response.status_code}: {response.text}")
            return False

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return False


def process_failed_uploads():
    """Retry failed uploads with exponential backoff."""
    if not failed_uploads:
        return

    # Process one failed upload per call
    image, timestamp, retry_count = failed_uploads.popleft()

    if retry_count >= MAX_RETRIES:
        logger.warning(f"Max retries reached for image {timestamp}, discarding")
        return

    logger.info(f"Retrying upload (attempt {retry_count + 1}/{MAX_RETRIES}) for {timestamp}")

    if upload_image(image, timestamp):
        logger.info(f"Retry successful for {timestamp}")
    else:
        # Re-queue with increased retry count
        failed_uploads.append((image, timestamp, retry_count + 1))
        logger.warning(f"Retry failed for {timestamp}, re-queued")


def capture_and_upload():
    """Main workflow: capture image and upload (or queue if fails)."""
    timestamp = datetime.now()
    logger.info("Starting capture cycle")

    image = capture_image()

    if image is None:
        logger.error("Skipping upload due to capture failure")
        return

    # Try to upload
    if upload_image(image, timestamp):
        logger.info("Capture and upload cycle completed successfully")
    else:
        # Queue for retry
        failed_uploads.append((image, timestamp, 0))
        logger.warning(f"Upload failed, queued for retry. Queue size: {len(failed_uploads)}")


def cleanup():
    """Cleanup resources on shutdown."""
    global camera
    if camera is not None:
        camera.stop()
        camera.close()
        logger.info("Camera closed")
    logger.info(f"Shutting down. {len(failed_uploads)} uploads still in queue.")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    logger.info("Interrupt received, shutting down...")
    cleanup()
    sys.exit(0)


def main():
    """Main scheduler loop."""
    # Load configuration first
    load_config()

    logger.info(f"Starting gauge monitor for device: {DEVICE_ID}")
    logger.info(f"Capture interval: {CAPTURE_INTERVAL} minutes")
    logger.info(f"Retry interval: {RETRY_INTERVAL} seconds")

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Schedule periodic capture every 5 minutes
    schedule.every(CAPTURE_INTERVAL).minutes.do(capture_and_upload)

    # Schedule retry processing every 30 seconds
    schedule.every(RETRY_INTERVAL).seconds.do(process_failed_uploads)

    # Run first capture immediately
    logger.info("Running initial capture")
    capture_and_upload()

    # Main loop
    logger.info("Entering scheduler loop")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
        cleanup()
        raise


if __name__ == '__main__':
    main()