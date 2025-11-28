from picamera2 import Picamera2
from PIL import Image
import time
from datetime import datetime
import logging

LOG_FILE = "get_testing_data.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = None

# Global camera instance
camera = None

# Configuration
CAPTURE_INTERVAL = 10  # seconds
OUTPUT_DIR = "captured_images"  # directory to save images


def initialize_camera():
    """Initialize the camera once at startup."""
    global camera
    
    try:
        camera = Picamera2()
        camera.configure(camera.create_still_configuration())
        camera.start()
        time.sleep(2)  # Camera warm-up
        logger.info("Camera initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Camera initialization failed: {e}")
        return False


def capture_image():
    """Capture image from Pi camera and return as PIL Image."""
    global camera

    try:
        # Capture image as numpy array and convert to PIL
        image_array = camera.capture_array()
        image = Image.fromarray(image_array)

        return image

    except Exception as e:
        logger.error(f"Camera capture failed: {e}")
        return None


def save_timestamped_image(image):
    """Save image with timestamp in filename."""
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gauge_{timestamp}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Save image
    image.save(filepath)
    logger.info(f"Saved timestamped image: {filepath}")
    
    return filepath


def capture_loop():
    """Main loop that captures images every 10 seconds."""
    
    # Initialize camera once
    if not initialize_camera():
        logger.error("Failed to initialize camera. Exiting.")
        return
    
    logger.info(f"Starting capture loop (every {CAPTURE_INTERVAL} seconds)")
    logger.info("Press Ctrl+C to stop")
    
    try:
        while True:
            # Capture image
            image = capture_image()
            
            if image is not None:
                # Optionally save with timestamp
                save_timestamped_image(image)
                
                # Here you could also process the image
                # For example, run your YOLO model:
                # results = model(image)
                
            else:
                logger.warning("Failed to capture image, will retry...")
            
            # Wait for next capture
            time.sleep(CAPTURE_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("\nCapture loop stopped by user")
    
    finally:
        # Cleanup
        if camera is not None:
            camera.stop()
            logger.info("Camera stopped")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    capture_loop()