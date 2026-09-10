"""RecycleVision AI -- Streamlit front end.

Presentation only. Every decision shown here is made in `recyclevision/`;
this file's job is to render a `SortResult` and let the user prod at it.
"""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image

from recyclevision import RoutingPolicy, SortingPipeline, annotate, vocabulary
from recyclevision.policy import PolicyError
from recyclevision.vocabulary import DEFAULT_VOCAB, Vocabulary

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_ROOT / "images"
POLICY_DIR = PROJECT_ROOT / "policies"

#: Sentinel for the closed-set COCO detector, kept as the measured baseline.
COCO = "coco"

st.set_page_config(page_title="RecycleVision AI", page_icon="♻️", layout="wide")


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model…")
def load_detector(kind: str):
    """Loaded once per detector. Swapping policy must not re-pay for a model."""
    from recyclevision.detector import OpenVocabularyDetector, YoloDetector
    from recyclevision.vocabulary import Vocabulary

    if kind == COCO:
        return YoloDetector()
    return OpenVocabularyDetector(Vocabulary.load(kind))


@st.cache_resource(show_spinner=False)
def load_policy(path: str) -> RoutingPolicy:
    return RoutingPolicy.load(path)


@st.cache_resource(show_spinner=False)
def discover_policies() -> list[tuple[Path, str]]:
    """Every policy on disk, paired with its declared name.

    The picker shows the name from inside the file rather than a prettified
    filename: "MRF Sorting Line" beats "Mrf Conveyor".
    """
    found = []
    for path in sorted(POLICY_DIR.glob("*.yaml")):
        try:
            found.append((path, RoutingPolicy.load(path).name))
        except PolicyError:
            # A broken policy should not take the whole app down; it simply
            # does not appear in the picker.
            continue
    return found


@st.cache_data(show_spinner=False)
def sort_image(_pipeline: SortingPipeline, image_bytes: bytes, policy_path: str, confidence: float):
    """Run the pipeline, keyed on the image, policy and threshold.

    Cached so that flipping a display toggle re-renders the annotation
    without paying for inference again.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image, _pipeline.sort(image, confidence=confidence)


policies = discover_policies()
if not policies:
    st.error(f"No usable routing policies found in {POLICY_DIR}.")
    st.stop()
policy_names = dict(policies)


def _vocabulary_order(path: Path) -> tuple[int, str]:
    """Put the project default first so the picker opens on the current best.

    Older vocabularies stay selectable — comparing against them is how a
    regression gets spotted — but they should not be what a visitor sees first.
    """
    return (0 if path == DEFAULT_VOCAB else 1, path.name)


detector_options: list[tuple[str, str]] = [
    (str(path), f"Open vocabulary · {Vocabulary.load(path).name}")
    for path in sorted(vocabulary.discover(), key=_vocabulary_order)
]
detector_options.append((COCO, "Stock COCO (baseline)"))
detector_labels = dict(detector_options)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    detector_kind = st.selectbox(
        "Detector",
        options=[key for key, _ in detector_options],
        format_func=lambda k: detector_labels[k],
        help=(
            "An open-vocabulary model is told what to look for in words, so it "
            "can name a drink can. COCO cannot."
        ),
    )

    try:
        detector = load_detector(detector_kind)
    except Exception as exc:  # noqa: BLE001 - surface anything, not just the logs
        st.error(f"Could not load the detector: {exc}")
        st.info("Check your network connection — model weights download on first run.")
        st.stop()

    weights = detector.weights

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.15,
        step=0.05,
        help="Lower catches more objects but with more false positives.",
    )

    st.subheader("Display")
    show_boxes = st.checkbox("Bounding boxes", value=True)
    show_labels = st.checkbox("Bin labels", value=True)
    show_confidence = st.checkbox("Confidence on labels", value=True)

    st.divider()

    st.subheader("Model")
    if weights.is_custom:
        st.success(weights.display_name)
    else:
        st.warning(weights.display_name)
        st.caption(weights.caveat)

    st.subheader("Policy")
    policy_path = st.selectbox(
        "Routing rules",
        options=[path for path, _ in policies],
        format_func=lambda p: policy_names[p],
        help="Local recycling rules. Same detector, different destinations.",
        label_visibility="collapsed",
    )

    try:
        policy = load_policy(str(policy_path))
    except PolicyError as exc:
        st.error(f"{policy_path.name}: {exc}")
        st.stop()

    st.caption(policy.description)

    with st.expander("Bins in this policy"):
        for bin_ in policy.bins:
            st.markdown(f"**{bin_.name}**")
            st.caption(bin_.description)

pipeline = SortingPipeline(detector, policy)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.title("♻️ RecycleVision AI")
st.markdown("#### Point it at waste. It tells you which bin each item goes in — and why.")

if not weights.is_custom:
    st.warning(f"**Demo mode.** {weights.caveat}", icon="⚠️")


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

samples = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

uploaded = st.file_uploader(
    "Upload a photo of waste or recycling",
    type=["jpg", "jpeg", "png"],
)

if samples:
    st.caption("…or try one of the samples:")
    sample_cols = st.columns(min(len(samples), 4))
    for column, sample in zip(sample_cols, samples, strict=False):
        if column.button(sample.stem, use_container_width=True):
            st.session_state["sample"] = str(sample)

image_bytes: bytes | None = None
source_name = ""

if uploaded is not None:
    image_bytes = uploaded.getvalue()
    source_name = uploaded.name
    st.session_state.pop("sample", None)
elif "sample" in st.session_state:
    sample_path = Path(st.session_state["sample"])
    if sample_path.is_file():
        image_bytes = sample_path.read_bytes()
        source_name = sample_path.name


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def render_bin_card(bin_, items) -> None:
    """One bin, its items, and what to do with them."""
    with st.container(border=True):
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem'>"
            f"<span style='width:1rem;height:1rem;border-radius:3px;"
            f"background:{bin_.color};display:inline-block'></span>"
            f"<strong style='font-size:1.05rem'>{bin_.name}</strong>"
            f"<span style='margin-left:auto;opacity:.65'>{len(items)}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(bin_.description)

        for item in items:
            flag = " ⚠️" if item.needs_review else ""
            st.markdown(f"**{item.label}**{flag} · {item.confidence:.0%}")
            if item.handling:
                st.caption(item.handling)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

if image_bytes is not None:
    with st.spinner("Analysing…"):
        image, result = sort_image(pipeline, image_bytes, str(policy_path), confidence)
        annotated = annotate(
            image,
            result,
            show_boxes=show_boxes,
            show_labels=show_labels,
            show_confidence=show_confidence,
        )

    if result.total_items == 0:
        st.info(
            "No waste items detected. Try lowering the confidence threshold in the "
            "sidebar, or use a photo where the items are clearly separated."
        )
        if result.ignored:
            seen = sorted({d.label for d in result.ignored})
            st.caption(f"The model did see: {', '.join(seen)} — none of which are waste.")

    # ---- headline metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Items", result.total_items)
    m2.metric(
        "Diverted from landfill",
        f"{result.diversion_rate:.0%}",
        help="Share of items headed anywhere other than landfill.",
    )
    m3.metric(
        "Contamination",
        f"{result.contamination_rate:.0%}",
        help="Share of the stream that is not recoverable. The metric sorting facilities track.",
    )
    m4.metric(
        "Needs review",
        len(result.items_for_review),
        help="Items where the policy cannot be sure of the destination.",
    )

    st.divider()

    # ---- images
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Sorted")
        st.image(annotated, use_container_width=True)
        st.caption("Each box is coloured by destination bin, not by object class.")

    # ---- bins
    if result.items:
        st.divider()
        st.subheader("Where it goes")

        bins_used = result.bins_used
        for row_start in range(0, len(bins_used), 3):
            row = bins_used[row_start : row_start + 3]
            columns = st.columns(3)
            for column, bin_ in zip(columns, row, strict=False):
                with column:
                    render_bin_card(bin_, result.items_in(bin_.key))

    # ---- review queue
    if result.items_for_review:
        st.divider()
        st.subheader("⚠️ Worth a second look")
        st.caption(
            "The model is confident about what it saw; the policy is not confident "
            "about where it goes."
        )
        for item in result.items_for_review:
            with st.container(border=True):
                st.markdown(f"**{item.label}** → {item.bin.name}")
                if item.rationale:
                    st.write(item.rationale)

    # ---- details
    with st.expander("Detection details"):
        st.caption(f"Model: {result.model_name} · Policy: {result.policy_name} · {source_name}")

        if result.items:
            st.dataframe(
                [
                    {
                        "Item": item.label,
                        "Detected as": item.detection.label,
                        "Bin": item.bin.name,
                        "Material": item.material,
                        "Confidence": f"{item.confidence:.1%}",
                        "Certainty": item.certainty.value,
                    }
                    for item in result.items
                ],
                use_container_width=True,
                hide_index=True,
            )

        if result.ignored:
            seen = sorted({d.label for d in result.ignored})
            st.caption(
                f"Ignored as non-waste: {', '.join(seen)}. "
                "These are excluded from every metric above."
            )

else:
    # ---------------------------------------------------------------- empty
    st.info("Upload an image or pick a sample to begin.")
    st.divider()

    st.subheader("How it works")
    step1, step2, step3 = st.columns(3)
    with step1:
        st.markdown("### 1️⃣ Detect")
        st.write("A vision model locates every object in the image.")
    with step2:
        st.markdown("### 2️⃣ Route")
        st.write(
            "A routing policy decides which bin each item belongs in — "
            "based on local rules, not just what it is made of."
        )
    with step3:
        st.markdown("### 3️⃣ Explain")
        st.write(
            "Every decision comes with handling instructions, and flags "
            "the items a human should check."
        )

    st.divider()
    st.subheader("Why bins, not materials")
    st.markdown(
        """
Material does not determine destination, which is why classifying waste by
material gets the hard cases wrong:

- A **wine glass** is glass, but belongs in **landfill** — drinking glass melts at a
  different temperature and ruins a batch of recycled container glass.
- A **disposable coffee cup** is paper, but belongs in **landfill** — it is plastic-lined.
- A **pizza slice** is organic, and belongs in **compost**, not recycling.

RecycleVision routes to bins directly, and tells you why.
        """
    )


st.divider()
st.caption(f"RecycleVision AI v0.3 · {policy.name} · {weights.display_name}")
