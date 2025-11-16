from dotenv import load_dotenv
import os
from pathlib import Path
from ultralytics import YOLO


# load .env file
load_dotenv(override=True)


BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_BAS_PATH = os.path.join(BASE_PATH, "fighter-jet-yolo-OD-1")
MODEL_PATH = os.path.join(BASE_PATH, "runs/detect/train/weights/best.pt")







class Config:
    """Configuration class to hold application settings."""
    APP_NAME: str = os.getenv("APP_NAME")
    VERSION: str = os.getenv("VERSION")
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY")

    upload_folder = Path(BASE_PATH) / "uploads"
    output_folder = Path(BASE_PATH) / "runs/detect/production/outputs"  

    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_image_extensions: list = [".jpg", ".jpeg", ".png", ".bmp"]
    allowed_video_extensions: list = [".mp4", ".avi", ".mov", ".mkv"]

    # Load the model
    model = YOLO(MODEL_PATH)




settings = Config()

# Create two folders for uploads and outputs if they don't exist

settings.upload_folder.mkdir(parents=True, exist_ok=True)
settings.output_folder.mkdir(parents=True, exist_ok=True)



