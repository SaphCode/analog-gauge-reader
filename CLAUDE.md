# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Preferences

When assisting with this project, please follow these guidelines:

1. **Incremental Changes**: Make only small, focused changes to code unless explicitly asked to do larger refactors. This helps with learning and understanding.

2. **Educational Approach**: The developer is learning - prioritize understanding over speed. Explain decisions and trade-offs.

3. **Planning First**: Before implementing features:
   - Create a plan outlining the approach
   - Identify decision points where multiple options exist
   - Present different implementation options with their advantages and disadvantages
   - Wait for confirmation before proceeding with implementation

4. **Discuss Trade-offs**: When there are multiple ways to solve a problem, present the options with pros/cons so informed decisions can be made.

## Project Overview

This is an analog gauge monitoring system that uses YOLO object detection to read measurements from analog gauges (e.g., pressure gauges, thermometers). The system consists of:
- **Edge Device**: Raspberry Pi Zero with camera module for image capture and pre-processing
- **Cloud Processing**: Google Cloud Run function for gauge detection and measurement calculation
- **Algorithm**: Detects gauge components (center, needle tip, min/max markers) and calculates readings through geometric angle analysis

## Core Architecture

### Three-Stage Processing Pipeline

1. **Image Capture & Pre-processing** (Raspberry Pi Zero)
   - Camera module captures gauge images on schedule or trigger
   - Pre-processing applied (cropping, resizing, brightness adjustment, etc.)
   - Sends POST request with image to cloud function endpoint
   - Minimal processing on Pi to conserve resources

2. **Detection Stage** (`models/gauge_detector.py`)
   - Uses YOLO model to detect 5 classes: `center`, `gauge`, `max`, `min`, `tip`
   - Returns highest-confidence detection for each class
   - Model weights stored in `models/yolo_best.pt`
   - Runs in cloud function (computationally intensive)

3. **Reading Stage** (`models/gauge_reader.py`)
   - Converts bounding boxes to center points
   - Calculates vectors from gauge center to components
   - Computes angles between needle tip and min/max markers
   - Uses linear interpolation: `measurement = (angle_tip / angle_range) * (max_val - min_val) + min_val`
   - Critical: Most gauges span >180° so the system uses reflex angles (360° - direct_angle)

### Cloud Function Endpoint

`google-cloud-functions/process-image/` contains a Google Cloud Run function that:
- Accepts POST requests with multipart/form-data image uploads from Raspberry Pi
- Validates image format (JPEG, PNG, GIF, WebP, BMP) and size (max 10MB)
- Currently logs image receipt for debugging
- TODO: Integration with gauge detection/reading models pending
- Will return gauge reading as JSON response to Pi

### Data Storage and Dashboard

4. **Firestore Database** (Cloud Storage)
   - Stores gauge readings with timestamp, measurement value, and metadata
   - Uses Firebase Admin SDK for Python (`firebase_admin`)
   - **Database Design**: Single `readings` collection for all devices
     - Each document represents one gauge reading from one device
     - Readings are filtered by `device_id` field
     - Allows querying across all devices or filtering to specific device
   - **Document Structure**:
     ```javascript
     {
       device_id: "pi-001",      // Which Raspberry Pi sent this reading
       timestamp: Timestamp,      // When the reading was taken
       measurement: 145.2,        // The gauge reading value
       unit: "PSI",              // Measurement unit
       location: "Tank A",       // Optional: physical location
       gauge_min: 0,             // Optional: gauge scale minimum
       gauge_max: 160,           // Optional: gauge scale maximum
       image_url: "gs://..."     // Optional: link to source image in Cloud Storage
     }
     ```
   - Requires Google Cloud service account credentials

5. **Dashboard** (`dashboard/`)
   - Basic visualization and monitoring interface
   - Fetches historical data from Firestore `readings` collection
   - Can filter by device or view all devices
   - TODO: Add real-time monitoring, charts, and alerts

## Key Dependencies

The project uses:
- `ultralytics` - YOLO v8 model for object detection
- `numpy` - Vector and geometric calculations
- `functions-framework` + `Flask` - Google Cloud Functions (in cloud function only)
- `firebase-admin` - Firebase Admin SDK for Firestore access
- `pandas` - Data manipulation for dashboard
- `pillow` - Image processing (PIL)

Root `requirements.txt` contains dependencies for local development and dashboard.

## Development Environment

### Setup
```bash
# Virtual environment is in .venv/
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (no root requirements.txt - dependencies in .venv)
pip install ultralytics numpy torch torchvision
```

### Testing Models Locally
The `models/model_testing.ipynb` notebook is used for interactive development and testing of detection/reading algorithms.

Test images are in `tests/`:
- `test_image.jpg` - Primary test image
- `test_image_2.jpg` - Secondary test image

### Running Detection and Reading
```python
from models.gauge_detector import GaugeDetector
from models.gauge_reader import GaugeReader

# Load model and detect
detector = GaugeDetector('models/yolo_best.pt')
detections = detector.predict('tests/test_image.jpg')

# Read measurement
reader = GaugeReader(min_value=0, max_value=160, unit="PSI")
measurement = reader.read_gauge(detections)
print(f"Reading: {measurement:.2f} PSI")
```

## Important Implementation Details

### Angle Calculation Bug Fix
In `gauge_reader.py:176`, the system uses reflex angles: `max_angle = 360 - min_max_angle`

This is intentional because most analog gauges sweep 270° not 180°. The comments around lines 163-176 explain the reasoning. Do not change this without testing on actual gauge images.

### Detection Class Requirements
`GaugeReader.read_gauge()` requires all 4 detections: `center`, `tip`, `min`, `max`. Missing any class will raise `ValueError`. The `gauge` class is detected but not used in calculations.

### Cloud Function Deployment
```bash
cd google-cloud-functions/process-image

# Deploy to Google Cloud Run
gcloud functions deploy process-image \
  --runtime python312 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point process_image
```

## Raspberry Pi Edge Device

### Hardware Setup
- **Device**: Raspberry Pi Zero (W or 2W recommended for WiFi)
- **Camera**: Pi Camera Module (v2 or HQ camera)
- **Power**: Consider power consumption for always-on deployment

### Pi Client Responsibilities
The Raspberry Pi client code should:
1. Capture images from camera module on schedule (e.g., every 5 minutes)
2. Apply pre-processing (resize, crop, adjust brightness/contrast if needed)
3. Send POST request to cloud function with multipart/form-data
4. Handle response with gauge reading
5. Log or store readings locally as backup
6. Handle network failures gracefully (retry logic, offline queue)

### Example Pi POST Request
```python
import requests

# Send image to cloud function
with open('gauge_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post(
        'https://your-cloud-function-url',
        files=files,
        timeout=30
    )

if response.status_code == 200:
    result = response.json()
    # Process gauge reading from result
```

### Pi Resource Constraints
- Pi Zero has limited CPU/RAM - avoid running YOLO models locally
- Pre-processing should be lightweight (PIL/Pillow for basic image ops)
- Consider image compression to reduce upload time/data costs
- Battery-powered deployments need power optimization (reduce capture frequency, sleep modes)

## File Organization

```
models/
  gauge_detector.py    # YOLO detection wrapper
  gauge_reader.py      # Geometric angle calculation
  yolo_best.pt         # Trained YOLO weights (6MB)
  model_testing.ipynb  # Development notebook

google-cloud-functions/
  process-image/
    main.py            # Cloud Run HTTP endpoint
    requirements.txt   # Cloud function dependencies only

tests/
  test_image.jpg       # Test images for development
  test_image_2.jpg

raspberry-pi/          # TODO: Pi client code (camera capture + POST)
```

## Security Notes

- `.env` files contain API keys and are gitignored
- Current `.env` in project root has `ANTHROPIC_API_KEY`
- Models directory also has its own `.env` and `.gitignore`
- Never commit credential files or model weights to version control
