"""Pneumonia X-ray classifier — Streamlit app with 3-model toggle.

Models (all binary sigmoid, P(pneumonia)):
  - Custom CNN  — 88% — model/model_cnn.keras
  - VGG16       — 94% — model/model_vgg16.keras
  - Xception    — 95% — model/model_xception.keras

Run with uv-managed venv:
  uv run streamlit run app.py
"""

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

MODEL_REGISTRY = {
    "Custom CNN — 88%": {
        "path": Path("model/model_cnn.keras"),
        "accuracy": 88,
        "description": "Lightweight custom CNN baseline.",
    },
    "VGG16 — 94%": {
        "path": Path("model/model_vgg16.keras"),
        "accuracy": 94,
        "description": "Transfer-learning VGG16 backbone + sigmoid head.",
    },
    "Xception — 95%": {
        "path": Path("model/model_xception.keras"),
        "accuracy": 95,
        "description": "Transfer-learning Xception backbone + sigmoid head.",
    },
}

CLASS_NAMES = {0: "Normal", 1: "Pneumonia"}

SAMPLE_IMAGES = {
    "Normal sample": {
        "path": Path("image/NORMAL2-IM-1436-0001.jpeg"),
        "expected": "Normal",
    },
    "Pneumonia sample": {
        "path": Path("image/person1946_bacteria_4874.jpeg"),
        "expected": "Pneumonia",
    },
}


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load a .keras model once and cache it. Lazy per selection."""
    import tensorflow as tf

    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image, model) -> np.ndarray:
    """Resize to model.input_shape, scale to [0,1], add batch dim."""
    # Auto-detect target size from the model (all are 224x224x3 here).
    _, h, w, _ = model.input_shape
    h = h or 224
    w = w or 224
    image = image.convert("RGB").resize((w, h))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_and_display(image: Image.Image, model, model_name: str, model_path: Path,
                          caption: str = "X-ray", expected_label: str | None = None) -> None:
    st.image(image, caption=caption, use_container_width=True)
    if expected_label:
        st.caption(f"Expected label (sample): **{expected_label}**")

    with st.spinner("Predicting..."):
        batch = preprocess_image(image, model)
        prob_pneumonia = float(model.predict(batch, verbose=0)[0][0])

    prob_normal = 1.0 - prob_pneumonia
    pred_class = 1 if prob_pneumonia >= 0.5 else 0
    confidence = prob_pneumonia if pred_class == 1 else prob_normal

    if pred_class == 1:
        st.error(f"### Prediction: {CLASS_NAMES[pred_class]} ({confidence:.1%})")
    else:
        st.success(f"### Prediction: {CLASS_NAMES[pred_class]} ({confidence:.1%})")

    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

    col1, col2 = st.columns(2)
    col1.metric("Pneumonia", f"{prob_pneumonia:.1%}")
    col2.metric("Normal", f"{prob_normal:.1%}")

    with st.expander("Details"):
        st.write(f"**Model:** {model_name} (`{model_path}`)")
        st.write(f"**Input shape:** `{model.input_shape}` (auto-detected)")
        st.write("**Preprocessing:** RGB → resize → scale to [0, 1]")
        st.write("**Threshold:** `p >= 0.5 → Pneumonia`, else Normal")

    st.warning("⚠️ Research demo only — not medical advice. Consult a clinician.")


def main() -> None:
    st.set_page_config(page_title="Pneumonia X-ray Classifier", page_icon="🫁", layout="centered")

    st.title("🫁 Pneumonia X-ray Classifier")
    st.write("Upload a chest X-ray image and cycle between the three trained models.")

    # --- Sidebar: model toggle ---
    with st.sidebar:
        st.header("Model selection")
        model_name = st.radio(
            "Choose a model",
            options=list(MODEL_REGISTRY.keys()),
            index=0,
            help="Cycle over the three models to compare predictions.",
        )
        info = MODEL_REGISTRY[model_name]
        st.metric("Reported accuracy", f"{info['accuracy']}%")
        st.caption(info["description"])
        st.divider()
        st.write("**Accuracy comparison**")
        st.bar_chart({k.split(" — ")[0]: v["accuracy"] for k, v in MODEL_REGISTRY.items()})

    model_path = MODEL_REGISTRY[model_name]["path"]
    if not model_path.exists():
        st.error(f"Model file not found: `{model_path}`")
        return

    with st.spinner(f"Loading {model_name}..."):
        try:
            model = load_model(str(model_path))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load model: {exc}")
            return

    # --- Main: image input (upload + built-in samples) ---
    uploaded = st.file_uploader(
        "Upload a chest X-ray", type=["jpg", "jpeg", "png"], help="X-ray image in JPG/PNG format."
    )

    if "sample_choice" not in st.session_state:
        st.session_state.sample_choice = None

    st.write("Or try a built-in sample:")
    sample_cols = st.columns(len(SAMPLE_IMAGES))
    for col, (label, cfg) in zip(sample_cols, SAMPLE_IMAGES.items()):
        with col:
            if cfg["path"].exists():
                col.image(str(cfg["path"]), caption=f"{label}\n(Expected: {cfg['expected']})",
                          use_container_width=True)
            else:
                col.warning(f"Missing: `{cfg['path']}`")
            if col.button(f"Use {label}", key=f"sample_{label}", use_container_width=True):
                st.session_state.sample_choice = label

    # Uploaded file takes priority; otherwise use the clicked sample.
    image, caption, expected = None, "", None
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            caption, expected = "Uploaded X-ray", None
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read image: {exc}")
            return
    elif st.session_state.sample_choice is not None:
        cfg = SAMPLE_IMAGES[st.session_state.sample_choice]
        if not cfg["path"].exists():
            st.error(f"Sample image not found: `{cfg['path']}`")
            return
        image = Image.open(cfg["path"])
        caption = st.session_state.sample_choice
        expected = cfg["expected"]

    if image is None:
        st.info("👆 Upload an image or click a sample above to get a prediction.")
        return

    predict_and_display(image, model, model_name, model_path, caption=caption,
                        expected_label=expected)


if __name__ == "__main__":
    main()
