import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1B2A;
    color: #E0E6ED;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #162032;
    border: 2px dashed #00BFA5;
    border-radius: 16px;
    padding: 1.5rem;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover { border-color: #00E5CC; }
[data-testid="stFileUploader"] label { color: #B0BEC5 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00BFA5, #0097A7);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.6rem 1.4rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Progress bars ── */
.stProgress > div > div { border-radius: 99px; }

/* ── Divider ── */
hr { border-color: #1E3A4A; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #162032;
    border: 1px solid #1E3A4A;
    border-radius: 14px;
    padding: 1rem 1.2rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0D1B2A; }
::-webkit-scrollbar-thumb { background: #00BFA5; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Tumor metadata ─────────────────────────────────────────────────────────────
TUMOR_INFO = {
    "glioma": {
        "label": "Glioma",
        "color": "#EF5350",
        "bg": "#1A0A0A",
        "border": "#EF5350",
        "icon": "🔴",
        "severity": "High Concern",
        "severity_color": "#EF5350",
        "description": (
            "Gliomas are tumors that originate in the glial cells of the brain or spine. "
            "They are the most common type of primary brain tumor, accounting for about 33% of all brain tumors."
        ),
        "symptoms": ["Persistent headaches", "Seizures", "Memory & cognitive changes", "Vision or speech problems", "Nausea and vomiting"],
        "treatment": ["Surgical resection", "Radiation therapy", "Chemotherapy (Temozolomide)", "Targeted therapy"],
        "note": "⚠️ Immediate consultation with a neuro-oncologist is strongly recommended.",
        "model_accuracy": "92%",
    },
    "meningioma": {
        "label": "Meningioma",
        "color": "#FFA726",
        "bg": "#1A1000",
        "border": "#FFA726",
        "icon": "🟠",
        "severity": "Moderate Concern",
        "severity_color": "#FFA726",
        "description": (
            "Meningiomas arise from the meninges — the membranes surrounding the brain and spinal cord. "
            "Most are benign and slow-growing, though some can be aggressive."
        ),
        "symptoms": ["Gradual headaches", "Hearing or vision loss", "Memory difficulties", "Weakness in limbs", "Personality changes"],
        "treatment": ["Active surveillance (watch & wait)", "Surgical removal", "Stereotactic radiosurgery", "Radiation therapy"],
        "note": "ℹ️ Many meningiomas are benign. A neurologist can advise on the best monitoring plan.",
        "model_accuracy": "84%",
    },
    "notumor": {
        "label": "No Tumor Detected",
        "color": "#66BB6A",
        "bg": "#0A1A0A",
        "border": "#66BB6A",
        "icon": "🟢",
        "severity": "All Clear",
        "severity_color": "#66BB6A",
        "description": (
            "No tumor was detected in this MRI scan. The brain tissue appears within normal classification parameters. "
            "Regular check-ups are still advised if symptoms persist."
        ),
        "symptoms": ["N/A — no tumor indicators found"],
        "treatment": ["No immediate treatment needed", "Continue regular health check-ups", "Consult a doctor if symptoms persist"],
        "note": "✅ Great news! Always follow up with a medical professional for a complete diagnosis.",
        "model_accuracy": "95%",
    },
    "pituitary": {
        "label": "Pituitary Tumor",
        "color": "#42A5F5",
        "bg": "#0A0F1A",
        "border": "#42A5F5",
        "icon": "🔵",
        "severity": "Requires Attention",
        "severity_color": "#42A5F5",
        "description": (
            "Pituitary tumors develop in the pituitary gland at the base of the brain. "
            "Most are benign adenomas that grow slowly, but they can disrupt hormone production significantly."
        ),
        "symptoms": ["Headaches behind the eyes", "Vision problems", "Hormonal imbalances", "Unexplained weight changes", "Fatigue and mood swings"],
        "treatment": ["Medication (dopamine agonists)", "Transsphenoidal surgery", "Radiation therapy", "Hormone replacement therapy"],
        "note": "ℹ️ Most pituitary tumors are non-cancerous but require endocrinology follow-up.",
        "model_accuracy": "99%",
    },
}

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tumor_model.keras")

model = load_model()
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1.5rem 0;">
    <div style="font-size: 3rem; margin-bottom: 0.4rem;">🧠</div>
    <h1 style="font-size: 2.6rem; font-weight: 800; color: #FFFFFF; margin: 0; letter-spacing: -1px;">
        Brain Tumor MRI <span style="color: #00BFA5;">Classifier</span>
    </h1>
    <p style="color: #78909C; font-size: 1.05rem; margin-top: 0.6rem; font-weight: 400;">
        AI-powered classification using EfficientNet deep learning &nbsp;•&nbsp; 4 tumor types &nbsp;•&nbsp; Real-time results
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Model Stats Row ───────────────────────────────────────────────────────────
st.markdown("""
<h3 style="color: #B0BEC5; font-size: 0.8rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 1rem;">
    Model Performance
</h3>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Overall Accuracy", "89%", help="Validation set accuracy across all 4 classes")
with col2:
    st.metric("Glioma Detection", "92%", help="Per-class accuracy for glioma")
with col3:
    st.metric("Meningioma", "84%", help="Per-class accuracy for meningioma")
with col4:
    st.metric("No Tumor", "95%", help="Per-class accuracy for no tumor")
with col5:
    st.metric("Pituitary", "99%", help="Per-class accuracy for pituitary")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Upload Section ────────────────────────────────────────────────────────────
st.markdown("""
<h3 style="color: #B0BEC5; font-size: 0.8rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.8rem;">
    Upload MRI Scan
</h3>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your MRI image here",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed",
)

# ── Inference & Results ───────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    img_resized = image.resize((300, 300))
    img_array  = np.array(img_resized)
    img_array  = preprocess_input(img_array)
    img_array  = np.expand_dims(img_array, axis=0)

    with st.spinner("Analysing MRI scan…"):
        prediction = model.predict(img_array, verbose=0)

    probs          = prediction[0]
    top_idx        = int(np.argmax(probs))
    predicted_key  = CLASS_NAMES[top_idx]
    confidence     = float(probs[top_idx]) * 100
    info           = TUMOR_INFO[predicted_key]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout: image | result card ────────────────────────────────
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.markdown("""
        <p style="color: #78909C; font-size: 0.75rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.5rem;">
            Uploaded Scan
        </p>
        """, unsafe_allow_html=True)
        st.image(image, use_column_width=True)

    with right:
        st.markdown(f"""
        <div style="
            background: {info['bg']};
            border: 2px solid {info['border']};
            border-radius: 20px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.2rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.6rem;">
                <span style="font-size: 2rem;">{info['icon']}</span>
                <div>
                    <div style="font-size: 0.72rem; font-weight: 600; letter-spacing: 2px; color: #78909C; text-transform: uppercase;">Prediction</div>
                    <div style="font-size: 1.9rem; font-weight: 800; color: {info['color']}; line-height: 1.1;">{info['label']}</div>
                </div>
                <div style="margin-left: auto; text-align: right;">
                    <span style="
                        background: {info['color']}22;
                        color: {info['color']};
                        border: 1px solid {info['color']}66;
                        border-radius: 99px;
                        padding: 0.3rem 1rem;
                        font-size: 0.8rem;
                        font-weight: 700;
                    ">{info['severity']}</span>
                </div>
            </div>

            <div style="margin: 1.2rem 0 0.4rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.35rem;">
                    <span style="color: #B0BEC5; font-size: 0.85rem; font-weight: 500;">Confidence</span>
                    <span style="color: {info['color']}; font-size: 1.1rem; font-weight: 800;">{confidence:.2f}%</span>
                </div>
                <div style="background: #1E3A4A; border-radius: 99px; height: 10px; overflow: hidden;">
                    <div style="
                        background: linear-gradient(90deg, {info['color']}99, {info['color']});
                        width: {confidence:.1f}%;
                        height: 100%;
                        border-radius: 99px;
                        transition: width 0.6s ease;
                    "></div>
                </div>
            </div>

            <p style="color: #90A4AE; font-size: 0.88rem; margin-top: 1.2rem; line-height: 1.65;">
                {info['description']}
            </p>

            <div style="
                background: {info['color']}11;
                border-left: 3px solid {info['color']};
                border-radius: 6px;
                padding: 0.7rem 1rem;
                margin-top: 1rem;
                color: #B0BEC5;
                font-size: 0.85rem;
            ">
                {info['note']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Detail section: symptoms | treatment | all-class probs ───────────────
    d1, d2, d3 = st.columns([1, 1, 1.2], gap="large")

    with d1:
        st.markdown(f"""
        <div style="background: #162032; border: 1px solid #1E3A4A; border-radius: 16px; padding: 1.4rem;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 2px; color: #78909C; text-transform: uppercase; margin-bottom: 1rem;">
                Common Symptoms
            </div>
            {"".join(f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0;border-bottom:1px solid #1E3A4A;color:#CFD8DC;font-size:0.88rem;"><span style="color:{info["color"]};">▸</span>{s}</div>' for s in info["symptoms"])}
        </div>
        """, unsafe_allow_html=True)

    with d2:
        st.markdown(f"""
        <div style="background: #162032; border: 1px solid #1E3A4A; border-radius: 16px; padding: 1.4rem;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 2px; color: #78909C; text-transform: uppercase; margin-bottom: 1rem;">
                Treatment Options
            </div>
            {"".join(f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.4rem 0;border-bottom:1px solid #1E3A4A;color:#CFD8DC;font-size:0.88rem;"><span style="color:#00BFA5;">✓</span>{t}</div>' for t in info["treatment"])}
        </div>
        """, unsafe_allow_html=True)

    with d3:
        st.markdown("""
        <div style="background: #162032; border: 1px solid #1E3A4A; border-radius: 16px; padding: 1.4rem;">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 2px; color: #78909C; text-transform: uppercase; margin-bottom: 1rem;">
                All Class Probabilities
            </div>
        """, unsafe_allow_html=True)

        bar_colors = {
            "glioma":     "#EF5350",
            "meningioma": "#FFA726",
            "notumor":    "#66BB6A",
            "pituitary":  "#42A5F5",
        }
        bar_labels = {
            "glioma":     "Glioma",
            "meningioma": "Meningioma",
            "notumor":    "No Tumor",
            "pituitary":  "Pituitary",
        }

        sorted_indices = np.argsort(probs)[::-1]
        for idx in sorted_indices:
            cls   = CLASS_NAMES[idx]
            pct   = float(probs[idx]) * 100
            color = bar_colors[cls]
            is_top = idx == top_idx
            label_style = f"font-weight: {'800' if is_top else '400'}; color: {'#FFFFFF' if is_top else '#90A4AE'};"
            st.markdown(f"""
            <div style="margin-bottom: 0.75rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="font-size: 0.85rem; {label_style}">{bar_labels[cls]}</span>
                    <span style="font-size: 0.85rem; color: {color}; font-weight: 700;">{pct:.1f}%</span>
                </div>
                <div style="background: #1E3A4A; border-radius: 99px; height: 8px; overflow: hidden;">
                    <div style="
                        background: {'linear-gradient(90deg, ' + color + '88, ' + color + ')' if is_top else color + '55'};
                        width: {pct:.1f}%;
                        height: 100%;
                        border-radius: 99px;
                    "></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background: #0A1520;
        border: 1px solid #1E3A4A;
        border-radius: 12px;
        padding: 1rem 1.4rem;
        text-align: center;
        color: #546E7A;
        font-size: 0.82rem;
    ">
        ⚕️ <strong style="color: #78909C;">Medical Disclaimer:</strong>
        This tool is for educational and research purposes only.
        It is <strong>not</strong> a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare provider.
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        color: #37474F;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🔬</div>
        <p style="font-size: 1.1rem; color: #546E7A; font-weight: 500;">
            Upload an MRI scan above to get an instant AI-powered classification.
        </p>
        <div style="display: flex; justify-content: center; gap: 1.2rem; flex-wrap: wrap; margin-top: 2rem;">
            <span style="background:#1A0A0A;border:1px solid #EF5350;color:#EF5350;padding:0.5rem 1.2rem;border-radius:99px;font-size:0.85rem;font-weight:600;">🔴 Glioma</span>
            <span style="background:#1A1000;border:1px solid #FFA726;color:#FFA726;padding:0.5rem 1.2rem;border-radius:99px;font-size:0.85rem;font-weight:600;">🟠 Meningioma</span>
            <span style="background:#0A1A0A;border:1px solid #66BB6A;color:#66BB6A;padding:0.5rem 1.2rem;border-radius:99px;font-size:0.85rem;font-weight:600;">🟢 No Tumor</span>
            <span style="background:#0A0F1A;border:1px solid #42A5F5;color:#42A5F5;padding:0.5rem 1.2rem;border-radius:99px;font-size:0.85rem;font-weight:600;">🔵 Pituitary</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
