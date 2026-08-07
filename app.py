from pathlib import Path

import streamlit as st
from PIL import Image

from vision.detector import WasteDetector


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RecycleVision AI",
    page_icon="♻️",
    layout="wide"
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pt"


# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_detector():
    return WasteDetector(MODEL_PATH)


detector = load_detector()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("⚙️ Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05
    )

    show_masks = st.checkbox(
        "Show Segmentation Masks",
        value=False,
        disabled=True
    )

    show_labels = st.checkbox(
        "Show Labels",
        value=True
    )

    show_boxes = st.checkbox(
        "Show Bounding Boxes",
        value=True
    )

    st.divider()

    st.header("Model")

    st.success("YOLOv8 Waste Detection")

    st.caption(
        "Prototype model trained to detect "
        "glass, metal, paper, plastic, and waste."
    )


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("♻️ RecycleVision AI")

st.markdown(
    """
### Intelligent Recycling Through Computer Vision

RecycleVision AI uses computer vision to identify recyclable
materials from images. This prototype demonstrates the
vision pipeline that will eventually analyze recycling
conveyor belts in real time.
"""
)

st.divider()


# --------------------------------------------------
# Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image of recyclable materials",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# Results
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Run detection
    with st.spinner("Analyzing image..."):

        result = detector.detect(
            image,
            confidence=confidence
        )

        counts = detector.get_counts(result)

        detections = detector.get_detections(result)

        annotated_image = detector.get_annotated_image(result)

    # --------------------------------------------------
    # Top Metrics
    # --------------------------------------------------

    total_objects = len(detections)

    if total_objects > 0:
        average_confidence = (
            sum(d["confidence"] for d in detections)
            / total_objects
        )
    else:
        average_confidence = 0

    recyclable_classes = {
        "plastic",
        "glass",
        "metal",
        "paper"
    }

    recyclable_objects = sum(
        count
        for material, count in counts.items()
        if material in recyclable_classes
    )

    recyclable_percentage = (
        recyclable_objects / total_objects * 100
        if total_objects > 0
        else 0
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Objects Detected",
        total_objects
    )

    metric2.metric(
        "Recyclable",
        f"{recyclable_percentage:.0f}%"
    )

    metric3.metric(
        "Avg. Confidence",
        f"{average_confidence:.0%}"
    )

    metric4.metric(
        "Model",
        "YOLOv8"
    )

    st.divider()

    # --------------------------------------------------
    # Images
    # --------------------------------------------------

    image_left, image_right = st.columns(2)

    with image_left:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    with image_right:

        st.subheader("AI Detection")

        st.image(
            annotated_image,
            use_container_width=True
        )

    st.divider()

    # --------------------------------------------------
    # Material Counts
    # --------------------------------------------------

    st.subheader("Material Counts")

    materials = [
        ("🟢 Plastic", "plastic"),
        ("🔵 Glass", "glass"),
        ("🟡 Metal", "metal"),
        ("🟤 Paper", "paper"),
        ("⚪ Waste", "waste"),
    ]

    columns = st.columns(len(materials))

    for column, (label, key) in zip(columns, materials):

        with column:

            count = counts.get(key, 0)

            st.metric(
                label,
                count
            )

    # --------------------------------------------------
    # Detection Details
    # --------------------------------------------------

    with st.expander("View Detection Details"):

        if detections:

            for detection in detections:

                st.write(
                    f"**{detection['class'].title()}** — "
                    f"{detection['confidence']:.1%} confidence"
                )

        else:

            st.write("No objects detected.")

else:

    # --------------------------------------------------
    # Empty State
    # --------------------------------------------------

    st.info(
        "Upload an image above to begin AI analysis."
    )

    st.divider()

    st.subheader("How It Works")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("### 1️⃣ Upload")
        st.write(
            "Upload an image containing recyclable "
            "materials."
        )

    with step2:
        st.markdown("### 2️⃣ Analyze")
        st.write(
            "The computer vision model identifies "
            "waste and recyclable materials."
        )

    with step3:
        st.markdown("### 3️⃣ Understand")
        st.write(
            "View detected objects, confidence scores, "
            "and material counts."
        )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "RecycleVision AI • Prototype v0.2"
)