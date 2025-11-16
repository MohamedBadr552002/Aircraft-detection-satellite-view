import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple
import time
from ultralytics import YOLO
from .schemas import Detection, DetectionResponse, DetectionResponses

class AircraftDetector:
    """Aircraft detection using YOLOv11"""
    
    def __init__(self, model: YOLO):
        """Initialize with a YOLO model instance
        
        Args:
            model: An initialized YOLO model instance
        """
        self.model = model
        self.class_names = self.model.names
        
    def detect_image(self, image_path: str) -> DetectionResponse:
        """Detect aircraft in a single image"""
        start_time = time.time()
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run inference
        results = self.model(image)
        detections = self._parse_results(results[0], image)
        
        processing_time = (time.time() - start_time) * 1000
        
        return DetectionResponse(
            filename=Path(image_path).name,
            detections=detections,
            total_detections=len(detections),
            processing_time_ms=processing_time
        )
    
    def annotate_and_save_image(self, image_path: str, output_folder: str) -> str:
        """Detect, annotate and save image"""
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run inference
        results = self.model(image)
        annotated_frame = self._annotate_frame(image.copy(), results[0])
        
        # Save annotated image
        output_path = Path(output_folder) / f"{Path(image_path).stem}_annotated.jpg"
        cv2.imwrite(str(output_path), annotated_frame)
        
        return str(output_path)
    
    def detect_video(self, video_path: str, output_path: str = None) -> Tuple[DetectionResponses, str]:
        """Detect aircraft in video and save annotated video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not read video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            start_time = time.time()
            
            # Run inference
            results = self.model(frame)
            detections = self._parse_results(results[0], frame)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Annotate frame
            annotated_frame = self._annotate_frame(frame.copy(), results[0])
            
            # Write to output video
            if output_path:
                out.write(annotated_frame)
            
            # Store detection response
            all_detections.append(DetectionResponse(
                filename=f"frame_{frame_count}",
                detections=detections,
                total_detections=len(detections),
                processing_time_ms=processing_time
            ))
        
        cap.release()
        if output_path:
            out.release()
        
        return DetectionResponses(
            predictions=all_detections,
            total_images=frame_count
        ), output_path
    
    def detect_batch(self, image_paths: List[str]) -> DetectionResponses:
        """Detect aircraft in multiple images"""
        all_detections = []
        
        for image_path in image_paths:
            try:
                detection = self.detect_image(image_path)
                all_detections.append(detection)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        return DetectionResponses(
            predictions=all_detections,
            total_images=len(all_detections)
        )
    
    def _parse_results(self, result, image) -> List[Detection]:
        """Parse YOLO results into Detection objects"""
        detections = []
        
        if result.boxes is None:
            return detections
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names[class_id]
            
            # Convert to center x, y, width, height
            x = float((x1 + x2) / 2)
            y = float((y1 + y2) / 2)
            width = float(x2 - x1)
            height = float(y2 - y1)
            
            detections.append(Detection(
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=confidence,
                class_name=class_name,
                class_id=class_id
            ))
        
        return detections
    
    def _annotate_frame(self, frame, result):
        """Draw bounding boxes on frame"""
        if result.boxes is None:
            return frame
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names[class_id]
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

