"""
Google Cloud Run Function for Image Processing

This function accepts image uploads via POST requests from Raspberry Pi edge devices.
Processes gauge images using YOLO detection and returns measurement readings.
"""

import logging
import os
import io
from typing import Tuple
from flask import Request, jsonify
import functions_framework
from PIL import Image
from google.cloud import storage, firestore
from gauge_detector import GaugeDetector
from gauge_reader import GaugeReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Allowed image MIME types
ALLOWED_MIME_TYPES = {
    'image/jpeg',
    'image/jpg', 
    'image/png',
    'image/gif',
    'image/webp',
    'image/bmp'
}

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Cloud Storage configuration
BUCKET_NAME = "analog-gauge-images"

# Initialize clients (module-level to reuse across requests)
# This happens once on cold start
detector = None
storage_client = None
firestore_client = None

def initialize_services():
    """Initialize gauge detector, storage, and firestore on cold start"""
    global detector, storage_client, firestore_client

    if detector is None:
        logger.info("Initializing gauge detector...")
        model_path = os.path.join(os.path.dirname(__file__), "yolo_best.pt")
        detector = GaugeDetector(model_path)
        logger.info("Detector initialized successfully")

    if storage_client is None:
        logger.info("Initializing Cloud Storage client...")
        storage_client = storage.Client()
        logger.info("Cloud Storage client initialized")

    if firestore_client is None:
        logger.info("Initializing Firestore client...")
        firestore_client = firestore.Client()
        logger.info("Firestore client initialized")


@functions_framework.http
def process_image(request: Request) -> Tuple[dict, int]:
    """
    Cloud Run function that accepts image uploads from Raspberry Pi edge devices.

    This function:
    - Only accepts POST requests
    - Accepts multipart form data with 'image' field
    - Accepts optional form fields: min_value, max_value, unit, timestamp
    - Detects gauge components and calculates reading
    - Returns gauge measurement as JSON

    Args:
        request (Request): Flask request object containing:
            - image: The gauge image file (required)
            - device_id: Unique identifier for the gauge/device (default: "default-gauge")
            - min_value: Minimum gauge value (default: 0)
            - max_value: Maximum gauge value (default: 100)
            - unit: Measurement unit (default: "units")
            - timestamp: Image capture timestamp in format YYYYMMDD_HHMMSS (optional)

    Returns:
        Tuple[dict, int]: JSON response and HTTP status code

    Example usage from Raspberry Pi:
        Method: POST
        URL: https://your-function-url
        Content-Type: multipart/form-data
        Body:
          - image: gauge_photo.jpg
          - device_id: gauge-1
          - min_value: 0
          - max_value: 160
          - unit: PSI
          - timestamp: 20250110_143022

    Example curl command:
        curl -X POST \
          -F "image=@gauge.jpg" \
          -F "device_id=gauge-1" \
          -F "min_value=0" \
          -F "max_value=160" \
          -F "unit=PSI" \
          -F "timestamp=20250110_143022" \
          https://your-function-url
    """

    # Initialize services on first request
    initialize_services()
    
    # Only accept POST requests
    if request.method != 'POST':
        logger.warning(f"Rejected {request.method} request from {request.remote_addr}")
        return (
            jsonify({
                'error': 'Method not allowed',
                'message': 'This endpoint only accepts POST requests'
            }),
            405
        )
    
    try:
        # Extract image from multipart form data
        if not request.files or 'image' not in request.files:
            logger.warning("No image file in request")
            return (
                jsonify({
                    'error': 'Bad request',
                    'message': 'No image file provided. Send as multipart form data with field name "image"'
                }),
                400
            )
        
        file = request.files['image']
        
        if file.filename == '':
            logger.warning("Empty filename in upload")
            return (
                jsonify({
                    'error': 'Bad request',
                    'message': 'No file selected'
                }),
                400
            )
        
        filename = file.filename
        content_type = file.content_type
        image_data = file.read()
        
        # Validate content type
        if content_type not in ALLOWED_MIME_TYPES:
            logger.warning(f"Invalid content type: {content_type}")
            return (
                jsonify({
                    'error': 'Unsupported media type',
                    'message': f'Content type {content_type} not supported. Allowed types: {", ".join(ALLOWED_MIME_TYPES)}'
                }),
                415
            )
        
        # Validate file size
        image_size = len(image_data)
        if image_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {image_size} bytes")
            return (
                jsonify({
                    'error': 'Payload too large',
                    'message': f'Image size ({image_size} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE} bytes)'
                }),
                413
            )
        
        # Extract gauge configuration from form data
        device_id = request.form.get('device_id', 'default-gauge')
        min_value = float(request.form.get('min_value', None))
        max_value = float(request.form.get('max_value', None))
        unit = request.form.get('unit', None)
        timestamp = request.form.get('timestamp', None)

        if device_id == 'default-gauge':
            logger.warning("The device id was not specified. Assuming default-gauge.")

        if not min_value:
            logger.error("No minimum gauge value supplied")
            return (
                jsonify({
                    'error': 'Bad request',
                    'message': 'Minimum gauge value is needed for reading the measurement'
                }),
                400
            )
    
        if not max_value:
            logger.error("No maximum gauge value supplied")
            return (
                jsonify({
                    'error': 'Bad request',
                    'message': 'Maximum gauge value is needed for reading the measurement'
                }),
                400
            )
        
        if not unit:
            logger.error("No unit given")
            return (
                jsonify({
                    'error': 'Bad request',
                    'message': 'Always give the unit, or you might crash a rocket some day.'
                }),
                400
            )
        
        if not timestamp:
            logger.error("No timestamp given.")
            return (
                jsonify({
                    'error': 'Bad request',
                    'message': 'Please include the time at which the picture was taken.'
                }),
                400
            )


        # Extract date from timestamp (first 8 chars: YYYYMMDD)
        date_folder = timestamp[:8] if timestamp and len(timestamp) >= 8 else None

        if not date_folder:
            logger.error("Invalid or missing timestamp")
            return (
                jsonify({
                    'error': 'Bad request',
                    'message': 'timestamp parameter required in format YYYYMMDD_HHMMSS'
                }),
                400
            )

        # Log request details
        logger.info("=" * 60)
        logger.info("GAUGE IMAGE RECEIVED")
        logger.info("=" * 60)
        logger.info(f"Device ID: {device_id}")
        logger.info(f"Filename: {filename}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Date: {date_folder}")
        logger.info(f"Content-Type: {content_type}")
        logger.info(f"Size: {image_size} bytes ({image_size / 1024:.2f} KB)")
        logger.info(f"Client IP: {request.remote_addr}")
        logger.info(f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}")
        logger.info(f"First 20 bytes (hex): {image_data[:20].hex()}")
        logger.info(f"Gauge Config: min={min_value}, max={max_value}, unit={unit}")
        logger.info("=" * 60)

        # Convert image data to PIL Image
        image = Image.open(io.BytesIO(image_data))

        # Run detection
        logger.info("Running gauge detection...")
        detections = detector.predict(image, verbose=False)

        # Create reader with gauge-specific configuration
        reader = GaugeReader(min_value=min_value, max_value=max_value, unit=unit)

        # Calculate measurement
        logger.info("Calculating gauge reading...")
        measurement = reader.read_gauge(detections, verbose=False)

        logger.info(f"Gauge reading: {measurement:.2f} {unit}")

        # Save image to Cloud Storage
        logger.info("Saving image to Cloud Storage...")
        image_path = f"{device_id}/{date_folder}/{timestamp}.jpg"
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(image_path)
        blob.upload_from_string(image_data, content_type='image/jpeg')
        logger.info(f"Image saved to: gs://{BUCKET_NAME}/{image_path}")

        # Save metadata to Firestore (measurement data only, not client info)
        logger.info("Saving metadata to Firestore...")
        doc_ref = firestore_client.collection('readings').add({
            'device_id': device_id,
            'timestamp': timestamp,
            'date': date_folder,
            'measurement': measurement,
            'unit': unit,
            'image_path': image_path,
            'detections_found': list(detections.keys()),
            'detections_confidence': {name: det.confidence for name, det in detections.items()},
            'created_at': firestore.SERVER_TIMESTAMP
        })
        firestore_doc_id = doc_ref[1].id
        logger.info(f"Metadata saved to Firestore with ID: {firestore_doc_id}")

        # Return success response with measurement and storage info
        response_data = {
            'status': 'success',
            'measurement': round(measurement, 2),
            'unit': unit,
            'timestamp': timestamp,
            'device_id': device_id,
            'storage': {
                'bucket': BUCKET_NAME,
                'image_path': image_path,
                'full_url': f"gs://{BUCKET_NAME}/{image_path}"
            },
            'firestore_id': firestore_doc_id,
            'detections': {
                'found': list(detections.keys()),
                'confidence': {name: round(det.confidence, 3) for name, det in detections.items()}
            }
        }

        logger.info(f"Returning success response")
        return (jsonify(response_data), 200)
    
    except Exception as e:
        # Log the full exception for debugging
        logger.error(f"Unexpected error processing request: {str(e)}", exc_info=True)
        
        return (
            jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred while processing your request'
            }),
            500
        )