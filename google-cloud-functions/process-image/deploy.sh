#!/bin/bash
set -e  # Exit on error
set -x  # Print each command before executing

echo "=========================================="
echo "Deploying Google Cloud Function"
echo "=========================================="

# Navigate to function directory
echo "Navigating to function directory..."
cd "$(dirname "$0")"
pwd

# Verify we're in the right place
echo "Current directory contents:"
ls -la

# Copy model files from ../../models/
echo ""
echo "Copying model files from ../../models/..."
echo "Checking if source files exist:"
ls -lh ../../models/gauge_detector.py ../../models/gauge_reader.py ../../models/yolo_best.pt

cp -v ../../models/gauge_detector.py .
cp -v ../../models/gauge_reader.py .
cp -v ../../models/yolo_best.pt .

echo ""
echo "Files copied. Current directory:"
ls -lh

# Deploy to Google Cloud
echo ""
echo "=========================================="
echo "Deploying to Google Cloud (europe-west3)..."
echo "=========================================="
gcloud functions deploy process-image \
  --gen2 \
  --region=europe-west3 \
  --runtime=python312 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point=process_image \
  --memory=2GB \
  --timeout=300s \
  --verbosity=info

# Clean up copied files (keeps function dir clean in git)
echo ""
echo "Cleaning up copied files..."
rm -v gauge_detector.py gauge_reader.py yolo_best.pt

echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
