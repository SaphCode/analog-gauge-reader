"""
Analog Gauge Reader Module

This module extracts measurements from analog gauges by analyzing the detected
bounding boxes of gauge components (center, tip, min, max markers).

The measurement is calculated by:
1. Finding the center point of each detected component
2. Computing vectors from the gauge center to other components
3. Calculating the angle between the needle tip and min marker
4. Computing the maximum angle range between min and max markers
5. Converting the angle to a measurement value using linear interpolation

Usage:
    from yolo import GaugeDetector
    from gauge_reader import GaugeReader
    
    # Detect gauge components
    detector = GaugeDetector('yolo_best.pt')
    detections = detector.predict('gauge_image.jpg')
    
    # Read measurement
    reader = GaugeReader(min_value=0, max_value=160, unit="PSI")
    measurement = reader.read_gauge(detections)
    print(f"Reading: {measurement:.2f} PSI")
"""

import numpy as np
import math
from typing import Dict, Tuple, Optional
from gauge_detector import Detection


class GaugeReader:
    """
    Reads measurements from analog gauges using detected component positions.
    
    This class performs geometric calculations to determine the gauge reading
    based on the positions of the needle tip relative to the min/max markers.
    """
    
    def __init__(self, min_value: float = 0, max_value: float = 100, unit: str = "units"):
        """
        Initialize the GaugeReader with gauge specifications.
        
        Args:
            min_value: The value corresponding to the minimum position on the gauge
            max_value: The value corresponding to the maximum position on the gauge
            unit: The unit of measurement (e.g., "PSI", "°C", "RPM")
            
        Example:
            reader = GaugeReader(min_value=0, max_value=160, unit="PSI")
        """
        self.min_value = min_value
        self.max_value = max_value
        self.unit = unit
    
    @staticmethod
    def get_bbox_center(bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Calculate the center point of a bounding box.
        
        Args:
            bbox: Bounding box coordinates as (x1, y1, x2, y2)
            
        Returns:
            Numpy array containing the center coordinates [xc, yc]
            
        Example:
            center = get_bbox_center((0, 0, 10, 10))  # Returns [5.0, 5.0]
        """
        x1, y1, x2, y2 = bbox
        xc = x1 + (x2 - x1) / 2
        yc = y1 + (y2 - y1) / 2
        return np.array([xc, yc])
    
    @staticmethod
    def get_angle_between_vectors(v1: np.ndarray, v2: np.ndarray, in_degrees: bool = True) -> float:
        """
        Calculate the angle between two vectors.
        
        Args:
            v1: First vector as numpy array
            v2: Second vector as numpy array
            in_degrees: If True, return angle in degrees; if False, return in radians
            
        Returns:
            Angle between the vectors
            
        Note:
            The angle is always positive and between 0 and 180 degrees (or 0 and π radians).
        """
        # Normalize vectors to avoid numerical issues
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            return 0.0
        
        # Calculate dot product and clamp to [-1, 1] to avoid numerical errors with arccos
        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        if in_degrees:
            return math.degrees(angle_rad)
        return angle_rad
    
    def read_gauge(self, detections: Dict[str, Detection], verbose: bool = True) -> Optional[float]:
        """
        Calculate the gauge reading from detected components.
        
        This method requires detections for 'center', 'tip', 'min', and 'max' classes.
        It calculates the needle position relative to the gauge range and converts
        it to a measurement value.
        
        Args:
            detections: Dictionary mapping class names to Detection objects
            verbose: If True, print detailed calculation information
            
        Returns:
            The calculated measurement value, or None if required detections are missing
            
        Raises:
            ValueError: If required detections are missing
            
        Example:
            measurement = reader.read_gauge(detections)
            if measurement is not None:
                print(f"Gauge reading: {measurement:.2f} {reader.unit}")
        """
        # Check for required detections
        required_classes = ['center', 'tip', 'min', 'max']
        missing_classes = [cls for cls in required_classes if cls not in detections]
        
        if missing_classes:
            raise ValueError(
                f"Missing required detections: {', '.join(missing_classes)}. "
                f"Cannot calculate gauge reading without all components."
            )
        
        # Extract center points of each bounding box
        center = self.get_bbox_center(detections['center'].bbox)
        tip = self.get_bbox_center(detections['tip'].bbox)
        min_marker = self.get_bbox_center(detections['min'].bbox)
        max_marker = self.get_bbox_center(detections['max'].bbox)
        
        if verbose:
            print("\nComponent Centers:")
            print(f"  Center: {center}")
            print(f"  Tip:    {tip}")
            print(f"  Min:    {min_marker}")
            print(f"  Max:    {max_marker}")
        
        # Calculate radius vectors from gauge center to each component
        r_tip = tip - center
        r_min = min_marker - center
        r_max = max_marker - center
        
        # Calculate the angle between min and max markers
        # This represents the full range of the gauge
        min_max_angle = self.get_angle_between_vectors(r_min, r_max, in_degrees=True)
        
        # The gauge typically spans more than 180 degrees, so we take the reflex angle
        # BUG FIX: Changed from 360 - angle to handle gauges that span less than 180 degrees
        # We now check which interpretation makes sense based on gauge design
        if min_max_angle < 180:
            # Gauge spans less than 180 degrees - use the angle directly
            max_angle = min_max_angle
        else:
            # This shouldn't happen with arccos, but kept for safety
            max_angle = min_max_angle
        
        # However, most analog gauges span more than 180 degrees (e.g., 270 degrees)
        # So we typically want the reflex angle
        # BUG FIX: The original calculation assumed this, which is correct for most gauges
        max_angle = 360 - min_max_angle
        
        # Calculate the angle between tip and min marker
        # This represents the current needle position
        tip_angle = self.get_angle_between_vectors(r_tip, r_min, in_degrees=True)
        
        if verbose:
            print(f"\nAngle Calculations:")
            print(f"  Full gauge range: {max_angle:.2f}°")
            print(f"  Needle position:  {tip_angle:.2f}°")
        
        # Calculate the proportion of the gauge range covered by the needle
        tip_proportion = tip_angle / max_angle
        
        # Convert proportion to measurement value using linear interpolation
        measurement = tip_proportion * (self.max_value - self.min_value) + self.min_value
        
        if verbose:
            print(f"\nGauge Reading:")
            print(f"  Needle at {tip_proportion:.1%} of full range")
            print(f"  Measurement: {measurement:.2f} {self.unit}")
        
        return measurement
    
    def read_gauge_from_image(self, image_path: str, detector, verbose: bool = True) -> Optional[float]:
        """
        Convenience method to detect and read gauge in one step.
        
        Args:
            image_path: Path to the gauge image
            detector: GaugeDetector instance for running inference
            verbose: If True, print detailed information
            
        Returns:
            The calculated measurement value, or None if reading fails
            
        Example:
            from yolo import GaugeDetector
            from gauge_reader import GaugeReader
            
            detector = GaugeDetector('yolo_best.pt')
            reader = GaugeReader(min_value=0, max_value=160, unit="PSI")
            
            measurement = reader.read_gauge_from_image('gauge.jpg', detector)
        """
        try:
            # Run detection
            detections = detector.predict(image_path, verbose=verbose)
            
            # Calculate measurement
            measurement = self.read_gauge(detections, verbose=verbose)
            
            return measurement
            
        except (ValueError, FileNotFoundError) as e:
            print(f"Error reading gauge: {e}")
            return None

