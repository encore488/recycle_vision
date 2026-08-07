# ♻️ RecycleVision AI

RecycleVision AI is a computer vision prototype designed to identify and classify recyclable materials from images.

The long-term goal is to develop a vision system capable of analyzing recycling conveyor belts in real time and eventually providing perception for automated robotic sorting.

## Current Prototype

Prototype 0.2 currently:

- Accepts uploaded images
- Detects waste using YOLOv8
- Identifies glass, metal, paper, plastic, and general waste
- Draws bounding boxes around detected objects
- Calculates material counts
- Displays confidence scores
- Provides an interactive Streamlit dashboard

## Tech Stack

- Python
- Streamlit
- Ultralytics YOLO
- PyTorch
- OpenCV

## Running Locally

Clone the repository:

```bash
git clone <repository-url>
cd recycle_sort
Create a virtual environment:

python -m venv .venv

Activate it:

Windows PowerShell:

.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Download the model weights and place them in:

models/best_model.pt

Run the application:

streamlit run app.py


Roadmap
 Streamlit prototype
 Image upload
 YOLO waste detection
 Material counting
 Custom recycling dataset
 Improved object detection
 Instance segmentation
 Video processing
 Real-time conveyor analysis
 Object tracking
 Edge deployment
 Robotic sorting integration
Project Status

Early prototype / active development.