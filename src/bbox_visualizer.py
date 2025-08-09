#!/usr/bin/env python3
"""
YOLO-style train number recognition result visualization module
Supports bounding box marking, confidence display, and multiple styles
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import colorsys
import random

class BboxVisualizer:
    """YOLO-style Bbox visualizer"""
    
    def __init__(self):
        """Initialize visualizer"""
        # Color configuration
        self.colors = {
            'success': (0, 255, 0),      # Green - successful recognition
            'low_confidence': (0, 165, 255),  # Orange - low confidence  
            'failed': (0, 0, 255),       # Red - recognition failed
            'text_bg': (0, 0, 0),        # Black - text background
            'text_color': (255, 255, 255) # White - text color
        }
        
        # Display configuration
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.box_thickness = 3
        
    def draw_bbox_on_image(self, image: np.ndarray, result: Dict, 
                          show_all_detections: bool = True,
                          confidence_threshold: float = 0.7) -> np.ndarray:
        """
        Draw YOLO-style recognition boxes on image
        
        Args:
            image: Original image
            result: OCR recognition result
            show_all_detections: Whether to show all detection results
            confidence_threshold: Confidence threshold
            
        Returns:
            Annotated image
        """
        # Copy image to avoid modifying original
        annotated_image = image.copy()
        
        # If no detection results, show "not detected"
        if not result.get('detected_texts'):
            self._draw_no_detection_message(annotated_image)
            return annotated_image
        
        # Draw all detected text boxes
        if show_all_detections:
            for detection in result['detected_texts']:
                self._draw_single_detection(annotated_image, detection, confidence_threshold)
        
        # Specially mark best recognition result
        if result.get('success') and result.get('train_number'):
            self._draw_best_result(annotated_image, result)
        
        # Add overall recognition status
        self._draw_status_info(annotated_image, result)
        
        return annotated_image
    
    def _draw_single_detection(self, image: np.ndarray, detection: Dict, 
                              confidence_threshold: float):
        """Draw single detection result"""
        bbox = detection['bbox']
        text = detection['text']
        confidence = detection['confidence']
        
        # Convert bbox format to OpenCV format
        points = np.array(bbox, dtype=np.int32)
        
        # Choose color based on confidence
        if confidence >= confidence_threshold:
            color = self.colors['success']
            label = f"{text} ({confidence:.3f})"
        else:
            color = self.colors['low_confidence'] 
            label = f"{text} ({confidence:.3f}) - Low"
        
        # Draw bounding box
        cv2.polylines(image, [points], True, color, self.box_thickness)
        
        # Calculate text position
        x_min = int(min(point[0] for point in bbox))
        y_min = int(min(point[1] for point in bbox))
        
        # Draw label background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, self.font, self.font_scale, self.font_thickness
        )
        
        cv2.rectangle(image, 
                     (x_min, y_min - text_height - 10),
                     (x_min + text_width, y_min),
                     color, -1)
        
        # Draw label text
        cv2.putText(image, label,
                   (x_min, y_min - 5),
                   self.font, self.font_scale,
                   self.colors['text_color'],
                   self.font_thickness)
    
    def _draw_best_result(self, image: np.ndarray, result: Dict):
        """Specially mark best recognition result"""
        train_number = result['train_number']
        confidence = result['confidence']
        
        # Find best result bbox
        best_bbox = None
        for detection in result['detected_texts']:
            if (detection['text'] == train_number and 
                detection['confidence'] == confidence):
                best_bbox = detection['bbox']
                break
        
        if best_bbox:
            points = np.array(best_bbox, dtype=np.int32)
            
            # Draw bold green border
            cv2.polylines(image, [points], True, self.colors['success'], 
                         self.box_thickness + 2)
            
            # Best result is already marked with bold green border
            # No additional text marker needed to avoid blocking the number
    
    def _draw_status_info(self, image: np.ndarray, result: Dict):
        """Draw overall recognition status information"""
        height, width = image.shape[:2]
        
        # Status information
        if result.get('success'):
            status_text = f"SUCCESS Train: {result['train_number']}"
            status_color = self.colors['success']
        else:
            status_text = "FAILED No Train Number"
            status_color = self.colors['failed']
        
        # Performance information
        processing_time = result.get('processing_time_ms', 0)
        perf_text = f"Time: {processing_time:.0f}ms"
        
        # Detection count
        num_detections = len(result.get('detected_texts', []))
        detection_text = f"Detections: {num_detections}"
        
        # Draw status background
        info_height = 80
        cv2.rectangle(image, (10, 10), (350, info_height), (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (350, info_height), status_color, 2)
        
        # Draw status text
        cv2.putText(image, status_text, (20, 35),
                   self.font, 0.6, status_color, 2)
        cv2.putText(image, perf_text, (20, 55),
                   self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(image, detection_text, (20, 70),
                   self.font, 0.5, (255, 255, 255), 1)
    
    def _draw_no_detection_message(self, image: np.ndarray):
        """Draw no detection message"""
        height, width = image.shape[:2]
        
        message = "No Train Number Detected"
        text_size = cv2.getTextSize(message, self.font, 1.0, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        
        # Warning text
        cv2.putText(image, message,
                   (text_x, text_y),
                   self.font, 1.0,
                   self.colors['failed'], 3)
    
    def save_annotated_image(self, image: np.ndarray, result: Dict, 
                           output_path: str):
        """Save annotated image"""
        annotated_image = self.draw_bbox_on_image(image, result)
        cv2.imwrite(output_path, annotated_image)
        return output_path
    
    def create_comparison_view(self, original_image: np.ndarray, 
                             result: Dict) -> np.ndarray:
        """Create comparison view of original and annotated images"""
        annotated_image = self.draw_bbox_on_image(original_image, result)
        
        # Horizontally concatenate original and annotated images
        height = max(original_image.shape[0], annotated_image.shape[0])
        
        # Adjust size to make height consistent
        if original_image.shape[0] != height:
            original_image = cv2.resize(original_image, 
                                      (original_image.shape[1], height))
        if annotated_image.shape[0] != height:
            annotated_image = cv2.resize(annotated_image,
                                       (annotated_image.shape[1], height))
        
        # Concatenate images
        comparison = np.hstack([original_image, annotated_image])
        
        # Add separator line
        mid_x = original_image.shape[1]
        cv2.line(comparison, (mid_x, 0), (mid_x, height), (255, 255, 255), 3)
        
        # Add titles with enhanced visibility at bottom corners
        title_font_scale = 1.2
        title_thickness = 3
        title_y = height - 20
        
        # Add background rectangles for better visibility
        original_text_size = cv2.getTextSize("Original", self.font, title_font_scale, title_thickness)[0]
        detected_text_size = cv2.getTextSize("Detected", self.font, title_font_scale, title_thickness)[0]
        
        # Background for "Original" title
        cv2.rectangle(comparison, (15, title_y - original_text_size[1] - 5), 
                     (25 + original_text_size[0], title_y + 5), (0, 0, 0), -1)
        cv2.rectangle(comparison, (15, title_y - original_text_size[1] - 5), 
                     (25 + original_text_size[0], title_y + 5), (255, 255, 255), 2)
        
        # Background for "Detected" title  
        detected_x = mid_x + 20
        cv2.rectangle(comparison, (detected_x - 5, title_y - detected_text_size[1] - 5), 
                     (detected_x + detected_text_size[0] + 5, title_y + 5), (0, 0, 0), -1)
        cv2.rectangle(comparison, (detected_x - 5, title_y - detected_text_size[1] - 5), 
                     (detected_x + detected_text_size[0] + 5, title_y + 5), (0, 255, 0), 2)
        
        # Add title text
        cv2.putText(comparison, "Original", (20, title_y),
                   self.font, title_font_scale, (255, 255, 255), title_thickness)
        cv2.putText(comparison, "Detected", (detected_x, title_y),
                   self.font, title_font_scale, (255, 255, 255), title_thickness)
        
        return comparison

def main():
    """Test visualization functionality"""
    print("=== YOLO-style Train Number Recognition Visualization Test ===")
    
    from train_number_ocr import TrainNumberOCR
    
    # Initialize
    ocr = TrainNumberOCR()
    visualizer = BboxVisualizer()
    
    # Test image
    image_path = 'raw/119car.jpg'
    image = cv2.imread(image_path)
    
    # Recognition
    result = ocr.recognize_train_number(image_path)
    
    # Visualization
    annotated_image = visualizer.draw_bbox_on_image(image, result)
    
    # Create comparison view
    comparison = visualizer.create_comparison_view(image, result)
    
    # Display results
    cv2.imshow('Original vs Annotated', comparison)
    cv2.imshow('Annotated Only', annotated_image)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save results
    cv2.imwrite('output/annotated_119car.jpg', annotated_image)
    cv2.imwrite('output/comparison_119car.jpg', comparison)
    print("Annotated images saved to output/ directory")

if __name__ == "__main__":
    main()
