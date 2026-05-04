"""
🌿 Crop Disease Detection — Streamlit App
Upload a leaf image and get an AI-powered disease diagnosis.
Uses a MobileNetV2 transfer-learning model trained on the PlantVillage dataset.
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import os
import json

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Class names (38 PlantVillage classes) ────────────────────────────────────
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join("models", "mobilenet_v2_model.h5")


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Global ─────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header gradient ────────────────────────────────────── */
.hero-header {
    background: linear-gradient(135deg, #0f9b58 0%, #34a853 40%, #4fc978 100%);
    border-radius: 16px;
    padding: 2.2rem 2rem 1.6rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(15, 155, 88, .25);
}
.hero-header h1 {
    color: #fff;
    font-size: 2.1rem;
    margin: 0 0 .3rem;
}
.hero-header p {
    color: rgba(255,255,255,.88);
    font-size: 1.05rem;
    margin: 0;
}

/* ── Result card ────────────────────────────────────────── */
.result-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1px solid #86efac;
    border-radius: 14px;
    padding: 1.6rem;
    margin-top: 1rem;
    box-shadow: 0 4px 14px rgba(34,197,94,.12);
}
.result-card.disease {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border-color: #fca5a5;
    box-shadow: 0 4px 14px rgba(239,68,68,.12);
}
.result-label {
    font-size: 1.45rem;
    font-weight: 700;
    margin-bottom: .3rem;
}
.result-conf {
    font-size: 1rem;
    color: #555;
}

/* ── Sidebar polish ─────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%);
}

/* ── Upload area ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Cached model loading ────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess(image: Image.Image):
    """Resize, convert to array, apply MobileNet preprocessing."""
    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    if arr.ndim == 2:  # grayscale → RGB
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:  # RGBA → RGB
        arr = arr[..., :3]
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def format_label(raw: str) -> tuple:
    """Return (crop, condition) from a class name like 'Tomato___Leaf_Mold'."""
    parts = raw.split("___")
    crop = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
    return crop, condition


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 About")
    st.markdown(
        "This app uses a **MobileNetV2** model fine-tuned on the "
        "[PlantVillage dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) "
        "to detect **38 crop disease classes** across 14 plant species."
    )
    st.divider()
    st.markdown("### 📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "95.31%")
    col2.metric("Loss", "0.1395")
    st.divider()
    st.markdown("### 🔗 Links")
    st.markdown(
        "- [GitHub Repo](https://github.com/TahaXCoder/crop-disease-detection)\n"
        "- [Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)"
    )
    st.divider()
    show_top_k = st.slider("Show top-K predictions", 1, 10, 5)

# ── Hero header ──────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero-header">
    <h1>🌿 Crop Disease Detection</h1>
    <p>Upload a leaf image to get an instant AI-powered diagnosis</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── File uploader ────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload a leaf image",
    help="Supported formats: JPG, JPEG, PNG, WEBP",
    accept_multiple_files=False,
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("❌ Could not read the file. Please upload a valid image (JPG, PNG, WEBP).")
        st.stop()

    # Show uploaded image
    st.image(image, caption="Uploaded leaf image", use_container_width=True)

    # Predict
    with st.spinner("🔍 Analyzing leaf …"):
        model = load_model()
        batch = preprocess(image)
        preds = model.predict(batch, verbose=0)[0]

    top_indices = preds.argsort()[::-1][:show_top_k]
    top_label = CLASS_NAMES[top_indices[0]]
    top_conf = float(preds[top_indices[0]])
    crop, condition = format_label(top_label)

    is_healthy = "healthy" in condition.lower()
    card_class = "" if is_healthy else "disease"
    emoji = "✅" if is_healthy else "⚠️"

    # ── Result card ──────────────────────────────────────────────────────
    st.markdown(
        f"""
<div class="result-card {card_class}">
    <div class="result-label">{emoji} {crop} — {condition}</div>
    <div class="result-conf">Confidence: {top_conf:.1%}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Top-K bar chart ──────────────────────────────────────────────────
    st.markdown("#### Top Predictions")
    import pandas as pd

    chart_data = pd.DataFrame(
        {
            "Class": [CLASS_NAMES[i].replace("___", " — ").replace("_", " ") for i in top_indices],
            "Confidence": [float(preds[i]) for i in top_indices],
        }
    )
    st.bar_chart(chart_data.set_index("Class"), horizontal=True, color="#34a853")

else:
    # Placeholder when nothing is uploaded
    st.info("👆 Upload a leaf photo above to get started.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with ❤️ using Streamlit · MobileNetV2 · PlantVillage Dataset")
