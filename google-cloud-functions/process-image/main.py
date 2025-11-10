"""
Google Cloud Run Function for Image Processing

This function accepts image uploads via POST requests from phone automation apps.
Currently configured to log image receipt for debugging purposes.
"""

import logging
from typing import Tuple
from flask import Request, jsonify
import functions_framework

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


@functions_framework.http
def process_image(request: Request) -> Tuple[dict, int]:
    """
    Cloud Run function that accepts image uploads from phone automation apps.
    
    This function:
    - Only accepts POST requests
    - Accepts multipart form data with 'image' field
    - Validates image format and size
    - Logs image receipt details for debugging
    - Returns appropriate success/error responses
    
    Args:
        request (Request): Flask request object containing the image data
        
    Returns:
        Tuple[dict, int]: JSON response and HTTP status code
        
    Example usage from phone automation app (like Tasker, Shortcuts, HTTP Shortcuts):
        Method: POST
        URL: https://your-function-url
        Content-Type: multipart/form-data
        Body: File field named "image" with your photo
        
    Example curl command:
        curl -X POST -F "image=@photo.jpg" https://your-function-url
    """
    
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
        
        # Log image receipt details (for debugging)
        logger.info("=" * 60)
        logger.info("IMAGE RECEIVED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Filename: {filename}")
        logger.info(f"Content-Type: {content_type}")
        logger.info(f"Size: {image_size} bytes ({image_size / 1024:.2f} KB)")
        logger.info(f"Client IP: {request.remote_addr}")
        logger.info(f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}")
        logger.info(f"First 20 bytes (hex): {image_data[:20].hex()}")
        logger.info("=" * 60)

        
        
        # TODO: Add your image processing logic here
        # For example:
        # - Image resizing
        # - Format conversion
        # - Object detection
        # - OCR
        # - Storage to Cloud Storage
        # processed_result = process_image_logic(image_data)
        
        # Return success response
        response_data = {
            'status': 'success',
            'message': 'Image received and logged successfully',
            'details': {
                'filename': filename,
                'content_type': content_type,
                'size_bytes': image_size,
                'size_kb': round(image_size / 1024, 2)
            }
        }
        
        logger.info(f"Returning success response: {response_data}")
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