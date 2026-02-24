import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────
# GLOBAL STYLING (PREMIUM DARK UI)
# ─────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0F172A, #0B1120) !important;
    color: #E2E8F0;
}

.stApp {
    background: transparent !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.hero {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #00F5A0, #00D9F5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #94A3B8;
    font-size: 1.05rem;
    font-weight: 500;
}

.glass {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 1.6rem;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 2px dashed #00F5A0;
    border-radius: 16px;
    padding: 1.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, #00F5A0, #00D9F5);
    color: #0F172A;
    border-radius: 12px;
    font-weight: 700;
    padding: 0.6rem 1.5rem;
    border: none;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 1rem;
    border: 1px solid rgba(255,255,255,0.06);
}

hr {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.1);
    margin: 2rem 0;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tumor_model.keras")

model = load_model()
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div style="font-size:3.5rem;">🧠</div>
    <div class="hero-title">Brain Tumor MRI Classifier</div>
    <div class="hero-sub">
        AI-powered diagnosis using EfficientNet • 4 Tumor Classes • Real-time Results
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Overall Accuracy", "89%")
c2.metric("Glioma", "92%")
c3.metric("Meningioma", "84%")
c4.metric("Pituitary", "99%")

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed"
)

# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((300, 300))
    img_array = np.array(img_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing MRI Scan..."):
        prediction = model.predict(img_array, verbose=0)

    probs = prediction[0]
    top_idx = np.argmax(probs)
    confidence = probs[top_idx] * 100
    predicted_class = CLASS_NAMES[top_idx]

    left, right = st.columns([1, 1.5])

    with left:
        st.image(image, use_column_width=True)

    with right:
        st.markdown(f"""
        <div class="glass">
            <h2 style="margin-top:0;">Prediction: {predicted_class.upper()}</h2>
            <p style="font-size:1.2rem;">Confidence: <b>{confidence:.2f}%</b></p>
            <div style="background:#1E293B; border-radius:10px; height:10px;">
                <div style="
                    background: linear-gradient(90deg,#00F5A0,#00D9F5);
                    width:{confidence}%;
                    height:100%;
                    border-radius:10px;">
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Class Probabilities")
    for i, cls in enumerate(CLASS_NAMES):
        st.progress(float(probs[i]))
        st.write(f"{cls.capitalize()}: {probs[i]*100:.2f}%")

else:
    st.markdown("""
    <div style="text-align:center; padding:3rem; color:#64748B;">
        Upload an MRI image above to receive AI-powered classification.
    </div>
    """, unsafe_allow_html=True)
