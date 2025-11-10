"""
YOLO Model Inference Module

This module provides functionality for loading a YOLO model and detecting
objects in images, specifically designed for analog gauge reading applications.

Classes detected:
    - center: Center point of the gauge
    - gauge: The entire gauge body
    - max: Maximum value marker on the gauge
    - min: Minimum value marker on the gauge
    - tip: Tip of the gauge needle

Usage:
    from yolo import GaugeDetector
    
    detector = GaugeDetector('path/to/yolo_best.pt')
    detections = detector.predict('path/to/image.jpg')
    
    # Access detections by class name
    center_box = detections['center']
    tip_box = detections['tip']
"""

from ultralytics import YOLO
from typing import Dict, Optional, Tuple
import os


class Detection:
    """
    Represents a single object detection with bounding box and metadata.
    
    Attributes:
        class_name (str): Name of the detected class
        confidence (float): Detection confidence score (0-1)
        bbox (Tuple[float, float, float, float]): Bounding box coordinates (x1, y1, x2, y2)
    """
    
    def __init__(self, class_name: str, confidence: float, bbox: Tuple[float, float, float, float]):
        """
        Initialize a Detection object.
        
        Args:
            class_name: Name of the detected class
            confidence: Confidence score between 0 and 1
            bbox: Bounding box as (x1, y1, x2, y2) coordinates
        """
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
    
    def __repr__(self) -> str:
        return (f"Detection(class='{self.class_name}', "
                f"confidence={self.confidence:.2%}, "
                f"bbox={[f'{x:.0f}' for x in self.bbox]})")


class GaugeDetector:
    """
    YOLO-based detector for analog gauge components.
    
    This class handles loading a YOLO model and running inference on images
    to detect gauge components (center, tip, min, max markers).
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the GaugeDetector with a trained YOLO model.
        
        Args:
            model_path: Path to the YOLO model weights file (.pt)
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
            ValueError: If the model cannot be loaded
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            print("=" * 60)
            print("MODEL LOADED")
            print("=" * 60)
            print(f"Classes: {self.class_names}")
            print(f"Number of classes: {len(self.class_names)}")
            print("=" * 60)
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
    
    def predict(self, image_path: str, verbose: bool = True) -> Dict[str, Detection]:
        """
        Run inference on an image and return the highest confidence detection for each class.
        
        This method detects all objects in the image and returns only the detection
        with the highest confidence score for each class. This is important for
        gauge reading where multiple false positives might occur.
        
        Args:
            image_path: Path to the input image
            verbose: If True, print detection information
            
        Returns:
            Dictionary mapping class names to Detection objects. Only includes
            classes that were detected in the image.
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            
        Example:
            detections = detector.predict('gauge.jpg')
            if 'tip' in detections:
                tip_bbox = detections['tip'].bbox
                confidence = detections['tip'].confidence
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Run inference
        results = self.model(image_path)
        
        # Dictionary to store the highest confidence detection for each class
        best_detections: Dict[str, Detection] = {}
        
        # Process all detections
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id]
            confidence = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Keep only the highest confidence detection for each class
            if class_name not in best_detections or confidence > best_detections[class_name].confidence:
                best_detections[class_name] = Detection(class_name, confidence, (x1, y1, x2, y2))
        
        if verbose:
            print(f"\nFound {len(best_detections)} unique class(es):")
            for i, (class_name, detection) in enumerate(best_detections.items(), 1):
                print(f"\nDetection {i}:")
                print(f"  Class: {detection.class_name}")
                print(f"  Confidence: {detection.confidence:.2%}")
                print(f"  Box: [{detection.bbox[0]:.0f}, {detection.bbox[1]:.0f}, "
                      f"{detection.bbox[2]:.0f}, {detection.bbox[3]:.0f}]")
        
        return best_detections
    
    def show_results(self, image_path: str) -> None:
        """
        Run inference and display the image with bounding boxes.
        
        Args:
            image_path: Path to the input image
        """
        results = self.model(image_path)
        results[0].show()