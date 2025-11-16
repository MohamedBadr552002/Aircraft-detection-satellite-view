from pydantic import BaseModel
from typing import List, Optional

class Detection(BaseModel):
    """Single detection object"""
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str
    class_id: int

class DetectionResponse(BaseModel):
    """Response for a single image/frame"""
    filename: str
    detections: List[Detection]
    total_detections: int
    processing_time_ms: float
    
class DetectionResponses(BaseModel):
    """Response for multiple images/frames"""
    predictions: List[DetectionResponse]
    total_images: int