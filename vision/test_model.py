from pathlib import Path
from ultralytics import YOLO

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# File paths
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"
IMAGE_PATH = PROJECT_ROOT / "images" / "recycl_test2.jpeg"

# Load model
model = YOLO(MODEL_PATH)

# Run inference
results = model.predict(
    IMAGE_PATH,
    conf=0.25
)

# Display result
results[0].show()