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
# DARK PREMIUM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0B1120 !important;
    color: #F1F5F9 !important;
}

.stApp {
    background-color: #0B1120 !important;
}

.block-container {
    padding-top: 2rem !important;
    max-width: 1200px;
}

/* Hide streamlit chrome */
#MainMenu, footer, header {visibility: hidden;}

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 1rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg,#00F5A0,#00D9F5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: #CBD5E1 !important;
    font-size: 1.05rem;
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    color: #F8FAFC !important;
}

/* Make ALL paragraph text visible */
p, li, span, div {
    color: #E2E8F0 !important;
}

/* Headings brighter */
h1, h2, h3, h4 {
    color: #FFFFFF !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.08);
    border: 2px dashed #00F5A0;
    border-radius: 16px;
    padding: 1.5rem;
}

/* HR */
hr {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.15);
    margin: 2rem 0;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TUMOR INFO
# ─────────────────────────────────────────
TUMOR_INFO = {
    "glioma": {
        "color": "#EF4444",
        "severity": "High Concern",
        "desc": "Gliomas originate from glial cells in the brain or spine and are among the most common primary brain tumors.",
        "symptoms": ["Persistent headaches","Seizures","Vision problems","Speech difficulty"],
        "treatment": ["Surgery","Radiation therapy","Chemotherapy"]
    },
    "meningioma": {
        "color": "#F59E0B",
        "severity": "Moderate Concern",
        "desc": "Meningiomas arise from membranes surrounding the brain and are often benign and slow growing.",
        "symptoms": ["Gradual headaches","Memory issues","Hearing loss"],
        "treatment": ["Monitoring","Surgery","Radiation"]
    },
    "notumor": {
        "color": "#22C55E",
        "severity": "All Clear",
        "desc": "No tumor detected. Brain structure appears normal under model classification.",
        "symptoms": ["No tumor indicators"],
        "treatment": ["Routine check-ups"]
    },
    "pituitary": {
        "color": "#3B82F6",
        "severity": "Requires Attention",
        "desc": "Pituitary tumors occur in the hormone-regulating gland at the base of the brain.",
        "symptoms": ["Hormonal imbalance","Vision disturbance","Headaches"],
        "treatment": ["Medication","Surgery","Hormone therapy"]
    }
}

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tumor_model.keras")

model = load_model()
CLASS_NAMES = ["glioma","meningioma","notumor","pituitary"]

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
# UPLOADER
# ─────────────────────────────────────────
uploaded_file = st.file_uploader("Upload MRI", type=["jpg","png","jpeg"], label_visibility="collapsed")

# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img = image.resize((300,300))
    arr = preprocess_input(np.array(img))
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("Analyzing MRI Scan..."):
        pred = model.predict(arr, verbose=0)

    probs = pred[0]
    idx = np.argmax(probs)
    cls = CLASS_NAMES[idx]
    conf = probs[idx] * 100
    info = TUMOR_INFO[cls]

    col1, col2 = st.columns([1,1.5])

    with col1:
        st.image(image, use_column_width=True)

    with col2:
        st.markdown(f"""
        <div class="glass">
            <h2 style="color:{info['color']};margin-top:0;">Prediction: {cls.upper()}</h2>
            <p><b>Confidence:</b> {conf:.2f}%</p>
            <p><b>Severity:</b> {info['severity']}</p>
            <p style="margin-top:1rem;">{info['desc']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("### Symptoms")
        for s in info["symptoms"]:
            st.write("•", s)

    with s2:
        st.markdown("### Treatment Options")
        for t in info["treatment"]:
            st.write("•", t)

else:
    st.markdown("""
    <div style="text-align:center; padding:4rem; color:#64748B;">
    Upload an MRI image above to receive AI-powered classification.
    </div>
    """, unsafe_allow_html=True)
