import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI · Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    background-color: #050C15 !important;
    color: #C8D8E8 !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #050C15; }
::-webkit-scrollbar-thumb { background: #1CFFD4; border-radius: 99px; }

/* ── File uploader — fix invisible text ── */
[data-testid="stFileUploader"] {
    background: #0A1628 !important;
    border: 2px dashed #1CFFD4 !important;
    border-radius: 18px !important;
    padding: 2.5rem 2rem !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00FFE0 !important;
    background: #0D1E38 !important;
    box-shadow: 0 0 40px rgba(28,255,212,0.08);
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] div {
    color: #A0C0D8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stFileUploader"] button {
    background: #1CFFD4 !important;
    color: #050C15 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    padding: 0.5rem 1.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stFileUploader"] button:hover {
    background: #00FFE0 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #0A1628 !important;
    border: 1px solid #112240 !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.4rem !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span {
    color: #6A8FAF !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #1CFFD4 !important; }

/* ── Plotly bg fix ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TUMOR DATA
# ─────────────────────────────────────────────────────────────────────────────
TUMOR_INFO = {
    "glioma": {
        "label":       "Glioma",
        "color":       "#FF4D6D",
        "glow":        "rgba(255,77,109,0.25)",
        "bg":          "linear-gradient(135deg, #1A0510 0%, #0D0818 100%)",
        "border":      "#FF4D6D",
        "icon":        "⬤",
        "severity":    "HIGH CONCERN",
        "sev_color":   "#FF4D6D",
        "sev_bg":      "rgba(255,77,109,0.12)",
        "model_acc":   92,
        "description": (
            "Gliomas originate in the glial (supportive) cells of the brain or spinal cord. "
            "They represent ~33% of all primary brain tumors and range from slow-growing "
            "low-grade forms to highly aggressive glioblastomas (GBM)."
        ),
        "symptoms":    [
            "Persistent, worsening headaches",
            "Seizures or convulsions",
            "Cognitive & memory decline",
            "Vision or speech impairment",
            "Nausea and vomiting",
            "Progressive weakness in limbs",
        ],
        "treatment":   [
            "Surgical resection (craniotomy)",
            "Radiation therapy",
            "Temozolomide chemotherapy",
            "Bevacizumab (targeted therapy)",
            "Clinical trial enrollment",
        ],
        "note":        "⚠️ Seek immediate consultation with a neuro-oncologist.",
        "prognosis":   "Variable — depends heavily on grade and molecular markers (IDH, MGMT).",
    },
    "meningioma": {
        "label":       "Meningioma",
        "color":       "#FFB347",
        "glow":        "rgba(255,179,71,0.22)",
        "bg":          "linear-gradient(135deg, #1A1005 0%, #100D05 100%)",
        "border":      "#FFB347",
        "icon":        "⬤",
        "severity":    "MODERATE CONCERN",
        "sev_color":   "#FFB347",
        "sev_bg":      "rgba(255,179,71,0.12)",
        "model_acc":   84,
        "description": (
            "Meningiomas arise from the meninges — the layered membranes enveloping the brain and spinal cord. "
            "~90% are benign (Grade I) and slow-growing, though location can cause significant neurological effects."
        ),
        "symptoms":    [
            "Gradual-onset headaches",
            "Hearing or vision loss",
            "Memory difficulties",
            "Unilateral limb weakness",
            "Personality or mood changes",
            "Seizures (in some cases)",
        ],
        "treatment":   [
            "Active surveillance (watch & wait)",
            "Stereotactic radiosurgery (Gamma Knife)",
            "Surgical excision",
            "Fractionated radiotherapy",
        ],
        "note":        "ℹ️ Most meningiomas are benign — a neurologist can advise on the best plan.",
        "prognosis":   "Generally favourable; Grade I rarely recurs after complete resection.",
    },
    "notumor": {
        "label":       "No Tumor Detected",
        "color":       "#00E5A0",
        "glow":        "rgba(0,229,160,0.20)",
        "bg":          "linear-gradient(135deg, #051A10 0%, #050D0A 100%)",
        "border":      "#00E5A0",
        "icon":        "⬤",
        "severity":    "ALL CLEAR",
        "sev_color":   "#00E5A0",
        "sev_bg":      "rgba(0,229,160,0.10)",
        "model_acc":   95,
        "description": (
            "No tumor was detected in this MRI scan. Brain tissue appears within normal classification parameters. "
            "If symptoms persist, consider follow-up imaging and clinical evaluation."
        ),
        "symptoms":    [
            "No tumor indicators detected",
            "Continue symptom monitoring",
            "Discuss findings with physician",
        ],
        "treatment":   [
            "No immediate intervention needed",
            "Routine annual MRI if high-risk",
            "Follow up with neurologist if symptomatic",
        ],
        "note":        "✅ Great result! Always confirm with a qualified clinician.",
        "prognosis":   "Excellent — no mass identified on this scan.",
    },
    "pituitary": {
        "label":       "Pituitary Tumor",
        "color":       "#5BA8FF",
        "glow":        "rgba(91,168,255,0.22)",
        "bg":          "linear-gradient(135deg, #050A1A 0%, #050C18 100%)",
        "border":      "#5BA8FF",
        "icon":        "⬤",
        "severity":    "REQUIRES ATTENTION",
        "sev_color":   "#5BA8FF",
        "sev_bg":      "rgba(91,168,255,0.12)",
        "model_acc":   99,
        "description": (
            "Pituitary tumors (adenomas) develop in the pituitary gland at the brain's base. "
            "Most are benign but can disrupt the hypothalamic–pituitary axis, causing wide-ranging hormonal effects."
        ),
        "symptoms":    [
            "Headaches behind/above eyes",
            "Bitemporal visual field loss",
            "Hormonal imbalances (cortisol, GH, TSH)",
            "Unexplained weight gain/loss",
            "Fatigue and mood disturbances",
            "Reproductive irregularities",
        ],
        "treatment":   [
            "Dopamine agonists (prolactinoma)",
            "Transsphenoidal surgery (TSSA)",
            "Stereotactic radiosurgery",
            "Hormone replacement therapy",
            "Somatostatin analogues (acromegaly)",
        ],
        "note":        "ℹ️ Mostly benign — endocrinology + neurosurgery co-management recommended.",
        "prognosis":   "Good for most adenomas; depends on size, type, and hormonal involvement.",
    },
}

CLASS_NAMES  = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_STATS  = {"Overall": 89, "Glioma": 92, "Meningioma": 84, "No Tumor": 95, "Pituitary": 99}
BAR_COLORS   = {"glioma": "#FF4D6D", "meningioma": "#FFB347", "notumor": "#00E5A0", "pituitary": "#5BA8FF"}
BAR_LABELS   = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No Tumor", "pituitary": "Pituitary"}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tumor_model.keras")

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(180deg, #0A1628 0%, #050C15 100%);
    border-bottom: 1px solid #0D2240;
    padding: 3.5rem 4rem 3rem 4rem;
    position: relative;
    overflow: hidden;
">
    <!-- Subtle grid pattern -->
    <div style="
        position: absolute; inset: 0;
        background-image: linear-gradient(rgba(28,255,212,0.03) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(28,255,212,0.03) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
    "></div>

    <!-- Glowing orb -->
    <div style="
        position: absolute; top: -60px; right: 120px;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(28,255,212,0.07) 0%, transparent 70%);
        pointer-events: none;
    "></div>

    <div style="position: relative; max-width: 1100px; margin: 0 auto;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <span style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                         color: #1CFFD4; text-transform: uppercase; font-family: 'Space Mono', monospace;">
                ◈ NeuroScan AI
            </span>
            <span style="width: 40px; height: 1px; background: #1CFFD4; display: inline-block;"></span>
            <span style="font-size: 0.72rem; color: #3A5A7A; font-family: 'Space Mono', monospace;">v2.0 · EfficientNet</span>
        </div>

        <h1 style="
            font-family: 'DM Sans', sans-serif;
            font-size: clamp(2.2rem, 5vw, 3.6rem);
            font-weight: 700;
            color: #FFFFFF;
            margin: 0 0 0.6rem 0;
            line-height: 1.1;
            letter-spacing: -0.02em;
        ">
            Brain Tumor<br>
            <span style="
                background: linear-gradient(90deg, #1CFFD4, #5BA8FF);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">MRI Classifier</span>
        </h1>

        <p style="
            color: #6A8FAF !important;
            font-size: 1.05rem;
            margin: 0;
            max-width: 520px;
            line-height: 1.6;
        ">
            Upload a brain MRI scan for instant AI-powered classification across
            four tumor categories with clinical context and confidence analysis.
        </p>

        <!-- Stat pills -->
        <div style="display: flex; gap: 1rem; margin-top: 2rem; flex-wrap: wrap;">
            <span style="background: rgba(28,255,212,0.08); border: 1px solid rgba(28,255,212,0.2);
                         color: #1CFFD4 !important; border-radius: 99px; padding: 0.4rem 1rem;
                         font-size: 0.8rem; font-weight: 600;">89% Overall Accuracy</span>
            <span style="background: rgba(91,168,255,0.08); border: 1px solid rgba(91,168,255,0.2);
                         color: #5BA8FF !important; border-radius: 99px; padding: 0.4rem 1rem;
                         font-size: 0.8rem; font-weight: 600;">4 Tumor Classes</span>
            <span style="background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
                         color: #A0C0D8 !important; border-radius: 99px; padding: 0.4rem 1rem;
                         font-size: 0.8rem; font-weight: 600;">Real-time Inference</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div style="max-width: 1200px; margin: 0 auto; padding: 2.5rem 3rem;">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERFORMANCE STATS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<p style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
           color: #3A5A7A !important; text-transform: uppercase;
           font-family: 'Space Mono', monospace; margin-bottom: 1rem;">
    ◈ Model Performance Metrics
</p>
""", unsafe_allow_html=True)

mc = st.columns(5)
metric_data = [
    ("Overall", "89%", "#1CFFD4"),
    ("Glioma", "92%", "#FF4D6D"),
    ("Meningioma", "84%", "#FFB347"),
    ("No Tumor", "95%", "#00E5A0"),
    ("Pituitary", "99%", "#5BA8FF"),
]
for col, (label, val, color) in zip(mc, metric_data):
    with col:
        st.markdown(f"""
        <div style="
            background: #0A1628;
            border: 1px solid #112240;
            border-top: 3px solid {color};
            border-radius: 14px;
            padding: 1.2rem 1.3rem;
            text-align: center;
        ">
            <div style="font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
                        text-transform: uppercase; color: #4A6A8A !important; margin-bottom: 0.4rem;">
                {label}
            </div>
            <div style="font-size: 2rem; font-weight: 800; color: {color} !important;
                        font-family: 'Space Mono', monospace;">
                {val}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ACCURACY BAR CHART
# ─────────────────────────────────────────────────────────────────────────────
fig_model = go.Figure()

labels_chart = ["Overall", "Glioma", "Meningioma", "No Tumor", "Pituitary"]
values_chart = [89, 92, 84, 95, 99]
colors_chart = ["#1CFFD4", "#FF4D6D", "#FFB347", "#00E5A0", "#5BA8FF"]

fig_model.add_trace(go.Bar(
    x=labels_chart,
    y=values_chart,
    marker=dict(
        color=colors_chart,
        cornerradius=8,
        line=dict(width=0),
    ),
    text=[f"{v}%" for v in values_chart],
    textposition="outside",
    textfont=dict(color="#C8D8E8", size=13, family="Space Mono"),
    hovertemplate="<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>",
))

fig_model.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,22,40,0.6)",
    height=280,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis=dict(
        tickfont=dict(color="#6A8FAF", size=12, family="DM Sans"),
        gridcolor="rgba(255,255,255,0)",
        showline=False,
        zeroline=False,
    ),
    yaxis=dict(
        range=[0, 112],
        tickfont=dict(color="#6A8FAF", size=11),
        gridcolor="rgba(255,255,255,0.04)",
        showline=False,
        zeroline=False,
        ticksuffix="%",
    ),
    showlegend=False,
    bargap=0.35,
)

st.markdown("""
<div style="background: #0A1628; border: 1px solid #112240; border-radius: 18px; padding: 1.5rem 1.5rem 0.5rem;">
    <p style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.15em; color: #3A5A7A !important;
              text-transform: uppercase; font-family: 'Space Mono', monospace; margin: 0 0 0.2rem 0;">
        Per-Class Accuracy Breakdown
    </p>
""", unsafe_allow_html=True)
st.plotly_chart(fig_model, use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<p style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
           color: #3A5A7A !important; text-transform: uppercase;
           font-family: 'Space Mono', monospace; margin-bottom: 1rem;">
    ◈ Upload MRI Scan
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "upload",
    type=["jpg", "png", "jpeg"],
    label_visibility="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    img_r = image.resize((300, 300))
    arr   = preprocess_input(np.array(img_r).astype(np.float32))
    arr   = np.expand_dims(arr, axis=0)

    with st.spinner("Running inference…"):
        pred = model.predict(arr, verbose=0)

    probs       = pred[0]
    top_idx     = int(np.argmax(probs))
    cls         = CLASS_NAMES[top_idx]
    conf        = float(probs[top_idx]) * 100
    info        = TUMOR_INFO[cls]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Image + Result hero ───────────────────────────────────────────────────
    img_col, res_col = st.columns([1, 1.7], gap="large")

    with img_col:
        st.markdown("""
        <p style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
                  color: #3A5A7A !important; text-transform: uppercase;
                  font-family: 'Space Mono', monospace; margin-bottom: 0.6rem;">
            ◈ Uploaded Scan
        </p>
        """, unsafe_allow_html=True)
        st.image(image, use_column_width=True)

    with res_col:
        # ── Hero prediction card ──────────────────────────────────────────────
        st.markdown(f"""
        <div style="
            background: {info['bg']};
            border: 1.5px solid {info['border']}55;
            border-radius: 22px;
            padding: 2rem 2.2rem;
            box-shadow: 0 0 60px {info['glow']};
            margin-bottom: 1rem;
        ">
            <!-- Top row -->
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1.4rem;">
                <div>
                    <div style="font-size: 0.68rem; font-weight: 700; letter-spacing: 0.2em;
                                color: #3A5A7A !important; text-transform: uppercase;
                                font-family: 'Space Mono', monospace; margin-bottom: 0.4rem;">
                        Classification Result
                    </div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: {info['color']} !important;
                                line-height: 1; letter-spacing: -0.02em; font-family: 'DM Sans', sans-serif;">
                        {info['label']}
                    </div>
                </div>
                <span style="
                    background: {info['sev_bg']};
                    color: {info['sev_color']} !important;
                    border: 1px solid {info['sev_color']}44;
                    border-radius: 99px;
                    padding: 0.35rem 1.1rem;
                    font-size: 0.7rem;
                    font-weight: 700;
                    letter-spacing: 0.1em;
                    font-family: 'Space Mono', monospace;
                    white-space: nowrap;
                ">{info['severity']}</span>
            </div>

            <!-- Confidence bar -->
            <div style="margin-bottom: 1.4rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="font-size: 0.8rem; font-weight: 600; color: #6A8FAF !important;">Confidence Score</span>
                    <span style="font-size: 1.05rem; font-weight: 800; color: {info['color']} !important;
                                 font-family: 'Space Mono', monospace;">{conf:.2f}%</span>
                </div>
                <div style="background: rgba(255,255,255,0.06); border-radius: 99px; height: 8px; overflow: hidden;">
                    <div style="
                        background: linear-gradient(90deg, {info['color']}77, {info['color']});
                        width: {conf:.1f}%;
                        height: 100%;
                        border-radius: 99px;
                        box-shadow: 0 0 12px {info['color']}66;
                    "></div>
                </div>
            </div>

            <!-- Description -->
            <p style="color: #8AAAC0 !important; font-size: 0.9rem; line-height: 1.7; margin-bottom: 1.2rem;">
                {info['description']}
            </p>

            <!-- Prognosis -->
            <div style="
                background: rgba(255,255,255,0.04);
                border-radius: 10px;
                padding: 0.7rem 1rem;
                margin-bottom: 1rem;
                display: flex;
                gap: 0.6rem;
                align-items: flex-start;
            ">
                <span style="color: {info['color']} !important; font-weight: 700; font-size: 0.8rem; white-space: nowrap;">Prognosis</span>
                <span style="color: #7A9AB8 !important; font-size: 0.82rem; line-height: 1.5;">{info['prognosis']}</span>
            </div>

            <!-- Clinical note -->
            <div style="
                border-left: 3px solid {info['color']};
                padding: 0.6rem 1rem;
                border-radius: 0 8px 8px 0;
                background: {info['sev_bg']};
                color: #A0C0D8 !important;
                font-size: 0.83rem;
                line-height: 1.5;
            ">
                {info['note']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── All class probabilities — Plotly horizontal bar ──────────────────────
    sorted_idx  = np.argsort(probs)
    cls_sorted  = [CLASS_NAMES[i] for i in sorted_idx]
    pct_sorted  = [float(probs[i]) * 100 for i in sorted_idx]
    col_sorted  = [BAR_COLORS[c] for c in cls_sorted]
    lbl_sorted  = [BAR_LABELS[c] for c in cls_sorted]

    fig_prob = go.Figure()
    fig_prob.add_trace(go.Bar(
        y=lbl_sorted,
        x=pct_sorted,
        orientation="h",
        marker=dict(
            color=[BAR_COLORS[c] if c == cls else BAR_COLORS[c] + "55" for c in cls_sorted],
            cornerradius=6,
            line=dict(width=0),
        ),
        text=[f"  {p:.1f}%" for p in pct_sorted],
        textposition="outside",
        textfont=dict(
            color=[BAR_COLORS[c] if c == cls else "#4A6A8A" for c in cls_sorted],
            size=13,
            family="Space Mono",
        ),
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.2f}%<extra></extra>",
    ))
    fig_prob.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis=dict(
            range=[0, 115],
            tickfont=dict(color="#4A6A8A", size=11),
            gridcolor="rgba(255,255,255,0.04)",
            showline=False,
            zeroline=False,
            ticksuffix="%",
        ),
        yaxis=dict(
            tickfont=dict(color="#A0C0D8", size=13, family="DM Sans"),
            gridcolor="rgba(255,255,255,0)",
            showline=False,
            zeroline=False,
        ),
        showlegend=False,
        bargap=0.35,
    )

    # ── Donut chart ───────────────────────────────────────────────────────────
    fig_donut = go.Figure(data=[go.Pie(
        labels=[BAR_LABELS[c] for c in CLASS_NAMES],
        values=[float(probs[i]) * 100 for i in range(4)],
        hole=0.65,
        marker=dict(
            colors=[BAR_COLORS[c] for c in CLASS_NAMES],
            line=dict(color="#050C15", width=3),
        ),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
        direction="clockwise",
        sort=False,
    )])
    fig_donut.add_annotation(
        text=f"<b>{conf:.1f}%</b>",
        x=0.5, y=0.55,
        font=dict(size=22, color=info["color"], family="Space Mono"),
        showarrow=False,
    )
    fig_donut.add_annotation(
        text="confidence",
        x=0.5, y=0.38,
        font=dict(size=11, color="#4A6A8A", family="DM Sans"),
        showarrow=False,
    )
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(
            font=dict(color="#7A9AB8", size=11, family="DM Sans"),
            bgcolor="rgba(0,0,0,0)",
            x=0.5, xanchor="center",
            y=-0.08,
            orientation="h",
        ),
    )

    # ── Charts row ────────────────────────────────────────────────────────────
    chart_l, chart_r = st.columns([1.6, 1], gap="large")

    with chart_l:
        st.markdown(f"""
        <div style="background: #0A1628; border: 1px solid #112240;
                    border-radius: 18px; padding: 1.5rem 1.5rem 0.5rem;">
            <p style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
                      color: #3A5A7A !important; text-transform: uppercase;
                      font-family: 'Space Mono', monospace; margin: 0 0 0.2rem 0;">
                ◈ All Class Probabilities
            </p>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_r:
        st.markdown("""
        <div style="background: #0A1628; border: 1px solid #112240;
                    border-radius: 18px; padding: 1.5rem 1.5rem 0.5rem;">
            <p style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
                      color: #3A5A7A !important; text-transform: uppercase;
                      font-family: 'Space Mono', monospace; margin: 0 0 0.2rem 0;">
                ◈ Distribution
            </p>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Symptoms + Treatment ──────────────────────────────────────────────────
    sym_col, treat_col = st.columns(2, gap="large")

    def detail_card(title, items, color, bullet):
        rows = "".join(f"""
        <div style="
            display: flex; align-items: center; gap: 0.8rem;
            padding: 0.65rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            color: #A0C0D8 !important;
            font-size: 0.88rem;
        ">
            <span style="color: {color} !important; font-weight: 700; flex-shrink:0;">{bullet}</span>
            {item}
        </div>
        """ for item in items)
        return f"""
        <div style="
            background: #0A1628;
            border: 1px solid #112240;
            border-top: 3px solid {color};
            border-radius: 18px;
            padding: 1.5rem 1.6rem;
            height: 100%;
        ">
            <p style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
                      color: #3A5A7A !important; text-transform: uppercase;
                      font-family: 'Space Mono', monospace; margin: 0 0 0.8rem 0;">
                ◈ {title}
            </p>
            {rows}
        </div>
        """

    with sym_col:
        st.markdown(
            detail_card("Common Symptoms", info["symptoms"], info["color"], "▸"),
            unsafe_allow_html=True
        )

    with treat_col:
        st.markdown(
            detail_card("Treatment Options", info["treatment"], "#1CFFD4", "✓"),
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: #0A1628;
        border: 1px solid #112240;
        border-radius: 14px;
        padding: 1rem 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    ">
        <span style="font-size: 1.4rem;">⚕️</span>
        <p style="margin: 0; font-size: 0.82rem; color: #4A6A8A !important; line-height: 1.5;">
            <strong style="color: #6A8FAF !important;">Medical Disclaimer:</strong>
            This tool is for educational and research purposes only and is
            <strong style="color: #6A8FAF !important;">not</strong> a substitute for professional
            medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider
            before making any medical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem;">
        <div style="font-size: 3.5rem; margin-bottom: 1rem;">🔬</div>
        <p style="font-size: 1.1rem; font-weight: 600; color: #4A6A8A !important;">
            Upload a brain MRI scan above to begin classification
        </p>
        <p style="font-size: 0.88rem; color: #2A4A6A !important;">
            Supported formats: JPG · PNG · JPEG
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 2.5rem;">
            <span style="background:#FF4D6D11; border:1px solid #FF4D6D44; color:#FF4D6D !important;
                         padding:0.5rem 1.3rem; border-radius:99px; font-size:0.85rem; font-weight:600;">
                ⬤ Glioma &nbsp; 92%
            </span>
            <span style="background:#FFB34711; border:1px solid #FFB34744; color:#FFB347 !important;
                         padding:0.5rem 1.3rem; border-radius:99px; font-size:0.85rem; font-weight:600;">
                ⬤ Meningioma &nbsp; 84%
            </span>
            <span style="background:#00E5A011; border:1px solid #00E5A044; color:#00E5A0 !important;
                         padding:0.5rem 1.3rem; border-radius:99px; font-size:0.85rem; font-weight:600;">
                ⬤ No Tumor &nbsp; 95%
            </span>
            <span style="background:#5BA8FF11; border:1px solid #5BA8FF44; color:#5BA8FF !important;
                         padding:0.5rem 1.3rem; border-radius:99px; font-size:0.85rem; font-weight:600;">
                ⬤ Pituitary &nbsp; 99%
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
