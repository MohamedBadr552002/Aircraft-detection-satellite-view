from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import json
from pathlib import Path
from datetime import datetime
import os

from src.inference import AircraftDetector
from src.schemas import DetectionResponse, DetectionResponses
from src.config import settings

# Initialize FastAPI app
app = FastAPI(
    title="Aircraft Detection API",
    description="YOLO-based aircraft detection from satellite imagery",
    version="1.0.0"
)

# Initialize detector
detector = AircraftDetector(settings.model)

# Mount static files
app.mount("/outputs", StaticFiles(directory=settings.output_folder), name="outputs")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Aircraft Detection API",
        "endpoints": {
            "detect_image": "/detect/image",
            "detect_video": "/detect/video",
            "detect_batch": "/detect/batch",
            "download_result": "/results/{filename}"
        }
    }


api_key_header = APIKeyHeader(name='X-API-Key')
async def verify_api_key(api_key: str=Depends(api_key_header)):
    if api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="You are not authorized to use this API")
    return api_key


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...),  api_key: str=Depends(verify_api_key)):
    """
    Detect aircraft in a single image
    Returns: Detection results and saves annotated image
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.allowed_image_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format. Allowed: {settings.allowed_image_extensions}"
            )
        
        # Save uploaded file
        upload_path = Path(settings.upload_folder) / file.filename
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run detection
        detection_response = detector.detect_image(str(upload_path))
        
        # Save annotated image
        annotated_image = detector.annotate_and_save_image(
            str(upload_path),
            settings.output_folder
        )
        
        # Save response as JSON
        response_filename = f"{Path(file.filename).stem}_detections.json"
        response_path = Path(settings.output_folder) / response_filename
        
        with open(response_path, "w") as f:
            json.dump(detection_response.dict(), f, indent=2)
        
        return {
            "status": "success",
            "detections": detection_response.dict(),
            "annotated_image": f"/outputs/{Path(annotated_image).name}",
            "detections_json": f"/outputs/{response_filename}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded file
        if upload_path.exists():
            upload_path.unlink()


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...),  api_key: str=Depends(verify_api_key)):
    """
    Detect aircraft in a video
    Returns: Detection results for all frames and saves annotated video
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.allowed_video_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid video format. Allowed: {settings.allowed_video_extensions}"
            )
        
        # Save uploaded file
        upload_path = Path(settings.upload_folder) / file.filename
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Run detection
        output_video = Path(settings.output_folder) / f"{Path(file.filename).stem}_annotated.mp4"
        detection_responses, _ = detector.detect_video(str(upload_path), str(output_video))
        
        # Save response as JSON
        response_filename = f"{Path(file.filename).stem}_detections.json"
        response_path = Path(settings.output_folder) / response_filename
        
        with open(response_path, "w") as f:
            json.dump(detection_responses.dict(), f, indent=2)
        
        return {
            "status": "success",
            "total_frames": detection_responses.total_images,
            "detections_summary": [
                {
                    "frame": d.filename,
                    "detections_count": d.total_detections,
                    "processing_time_ms": d.processing_time_ms
                }
                for d in detection_responses.predictions
            ],
            "annotated_video": f"/outputs/{output_video.name}",
            "detections_json": f"/outputs/{response_filename}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded file
        if upload_path.exists():
            upload_path.unlink()


@app.post("/detect/batch")
async def detect_batch(files: list[UploadFile] = File(...),  api_key: str=Depends(verify_api_key)):
    """
    Detect aircraft in multiple images
    Returns: Detection results for all images
    """
    try:
        uploaded_paths = []
        results = []
        
        for file in files:
            # Validate file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in settings.allowed_image_extensions:
                continue
            
            # Save uploaded file
            upload_path = Path(settings.upload_folder) / file.filename
            with open(upload_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_paths.append(upload_path)
        
        if not uploaded_paths:
            raise HTTPException(status_code=400, detail="No valid images provided")
        
        # Run batch detection
        batch_responses = detector.detect_batch([str(p) for p in uploaded_paths])
        
        # Save annotated images and responses
        for i, detection in enumerate(batch_responses.predictions):
            # Save annotated image
            annotated_image = detector.annotate_and_save_image(
                str(uploaded_paths[i]),
                settings.output_folder
            )
        
        # Save batch response as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response_filename = f"batch_detections_{timestamp}.json"
        response_path = Path(settings.output_folder) / response_filename
        
        with open(response_path, "w") as f:
            json.dump(batch_responses.dict(), f, indent=2)
        
        return {
            "status": "success",
            "total_images": batch_responses.total_images,
            "detections_summary": [
                {
                    "filename": d.filename,
                    "detections_count": d.total_detections,
                    "processing_time_ms": d.processing_time_ms
                }
                for d in batch_responses.predictions
            ],
            "detections_json": f"/outputs/{response_filename}"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up uploaded files
        for path in uploaded_paths:
            if path.exists():
                path.unlink()


@app.get("/results/{filename}")
async def download_result(filename: str,  api_key: str=Depends(verify_api_key)):
    """Download result file (image, video, or JSON)"""
    try:
        file_path = Path(settings.output_folder) / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(file_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "YOLOv11",
        "upload_folder": settings.upload_folder,
        "output_folder": settings.output_folder
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)