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
# CSS  (injected once, at the top — no HTML rendering issues)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #050C15 !important;
    color: #C8D8E8 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #050C15; }
::-webkit-scrollbar-thumb { background: #1CFFD4; border-radius: 99px; }

/* ── File uploader fix — all text visible ── */
[data-testid="stFileUploader"] {
    background: #0A1628 !important;
    border: 2px dashed #1CFFD4 !important;
    border-radius: 18px !important;
    padding: 2rem !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] *,
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #A0C0D8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * { color: #A0C0D8 !important; }
[data-testid="stFileUploader"] button {
    background: #1CFFD4 !important;
    color: #050C15 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #1CFFD4 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
TUMOR_INFO = {
    "glioma": {
        "label":    "Glioma",
        "color":    "#FF4D6D",
        "glow":     "rgba(255,77,109,0.18)",
        "severity": "HIGH CONCERN",
        "acc":      92,
        "desc":     "Gliomas originate in the glial cells of the brain or spinal cord. They represent ~33% of all primary brain tumors and range from slow-growing low-grade forms to highly aggressive glioblastomas (GBM).",
        "prognosis":"Variable — depends on grade and molecular markers (IDH, MGMT).",
        "note":     "⚠️ Seek immediate consultation with a neuro-oncologist.",
        "symptoms": ["Persistent worsening headaches","Seizures or convulsions","Cognitive & memory decline","Vision or speech impairment","Nausea and vomiting","Progressive limb weakness"],
        "treatment":["Surgical resection (craniotomy)","Radiation therapy","Temozolomide chemotherapy","Bevacizumab (targeted therapy)","Clinical trial enrollment"],
    },
    "meningioma": {
        "label":    "Meningioma",
        "color":    "#FFB347",
        "glow":     "rgba(255,179,71,0.16)",
        "severity": "MODERATE CONCERN",
        "acc":      84,
        "desc":     "Meningiomas arise from the meninges — membranes enveloping the brain and spinal cord. ~90% are benign (Grade I) and slow-growing, though location can cause significant neurological effects.",
        "prognosis":"Generally favourable; Grade I rarely recurs after complete resection.",
        "note":     "ℹ️ Most meningiomas are benign — a neurologist can advise on the best plan.",
        "symptoms": ["Gradual-onset headaches","Hearing or vision loss","Memory difficulties","Unilateral limb weakness","Personality or mood changes","Seizures (in some cases)"],
        "treatment":["Active surveillance (watch & wait)","Stereotactic radiosurgery","Surgical excision","Fractionated radiotherapy"],
    },
    "notumor": {
        "label":    "No Tumor Detected",
        "color":    "#00E5A0",
        "glow":     "rgba(0,229,160,0.15)",
        "severity": "ALL CLEAR",
        "acc":      95,
        "desc":     "No tumor was detected in this MRI scan. Brain tissue appears within normal classification parameters. If symptoms persist, consider follow-up imaging and clinical evaluation.",
        "prognosis":"Excellent — no mass identified on this scan.",
        "note":     "✅ Great result! Always confirm findings with a qualified clinician.",
        "symptoms": ["No tumor indicators detected","Continue symptom monitoring","Discuss findings with physician"],
        "treatment":["No immediate intervention needed","Routine annual MRI if high-risk","Follow up with neurologist if symptomatic"],
    },
    "pituitary": {
        "label":    "Pituitary Tumor",
        "color":    "#5BA8FF",
        "glow":     "rgba(91,168,255,0.18)",
        "severity": "REQUIRES ATTENTION",
        "acc":      99,
        "desc":     "Pituitary tumors (adenomas) develop in the pituitary gland at the brain's base. Most are benign but can disrupt the hypothalamic–pituitary axis, causing wide-ranging hormonal effects.",
        "prognosis":"Good for most adenomas; depends on size, type, and hormonal involvement.",
        "note":     "ℹ️ Mostly benign — endocrinology + neurosurgery co-management recommended.",
        "symptoms": ["Headaches behind the eyes","Bitemporal visual field loss","Hormonal imbalances","Unexplained weight changes","Fatigue and mood disturbances","Reproductive irregularities"],
        "treatment":["Dopamine agonists (prolactinoma)","Transsphenoidal surgery","Stereotactic radiosurgery","Hormone replacement therapy"],
    },
}

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
COLORS      = {k: v["color"] for k, v in TUMOR_INFO.items()}
LABELS      = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No Tumor", "pituitary": "Pituitary"}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tumor_model.keras")

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(180deg,#0A1628 0%,#050C15 100%);
            border-bottom:1px solid #0D2240;padding:3rem 4rem 2.5rem;position:relative;overflow:hidden;">
    <div style="position:absolute;inset:0;
        background-image:linear-gradient(rgba(28,255,212,0.03) 1px,transparent 1px),
                         linear-gradient(90deg,rgba(28,255,212,0.03) 1px,transparent 1px);
        background-size:40px 40px;pointer-events:none;"></div>
    <div style="position:relative;max-width:1100px;margin:0 auto;">
        <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.18em;
                    color:#1CFFD4;text-transform:uppercase;font-family:'Space Mono',monospace;
                    margin-bottom:0.8rem;">◈ NeuroScan AI &nbsp;·&nbsp; EfficientNet v2</div>
        <h1 style="font-family:'DM Sans',sans-serif;font-size:3.2rem;font-weight:700;
                   color:#FFFFFF;margin:0 0 0.6rem 0;line-height:1.1;letter-spacing:-0.02em;">
            Brain Tumor<br>
            <span style="background:linear-gradient(90deg,#1CFFD4,#5BA8FF);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                MRI Classifier
            </span>
        </h1>
        <p style="color:#6A8FAF;font-size:1rem;margin:0 0 1.8rem;max-width:500px;line-height:1.6;">
            Upload a brain MRI scan for instant AI-powered classification across
            four tumor categories with clinical context and confidence analysis.
        </p>
        <div style="display:flex;gap:0.8rem;flex-wrap:wrap;">
            <span style="background:rgba(28,255,212,0.08);border:1px solid rgba(28,255,212,0.25);
                         color:#1CFFD4;border-radius:99px;padding:0.35rem 1rem;font-size:0.78rem;font-weight:600;">
                89% Overall Accuracy
            </span>
            <span style="background:rgba(91,168,255,0.08);border:1px solid rgba(91,168,255,0.25);
                         color:#5BA8FF;border-radius:99px;padding:0.35rem 1rem;font-size:0.78rem;font-weight:600;">
                4 Tumor Classes
            </span>
            <span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
                         color:#A0C0D8;border-radius:99px;padding:0.35rem 1rem;font-size:0.78rem;font-weight:600;">
                Real-time Inference
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONTENT AREA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div style="max-width:1200px;margin:0 auto;padding:2.5rem 3rem;">', unsafe_allow_html=True)

# ── Section label helper ──────────────────────────────────────────────────────
def section_label(text):
    st.markdown(f"""
    <p style="font-size:0.7rem;font-weight:700;letter-spacing:0.18em;color:#3A5A7A;
              text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:0.9rem;">
        ◈ {text}
    </p>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL STATS — metric tiles
# ─────────────────────────────────────────────────────────────────────────────
section_label("Model Performance Metrics")

stat_cols = st.columns(5)
stats = [
    ("Overall",    "89%", "#1CFFD4"),
    ("Glioma",     "92%", "#FF4D6D"),
    ("Meningioma", "84%", "#FFB347"),
    ("No Tumor",   "95%", "#00E5A0"),
    ("Pituitary",  "99%", "#5BA8FF"),
]
for col, (label, val, color) in zip(stat_cols, stats):
    with col:
        st.markdown(f"""
        <div style="background:#0A1628;border:1px solid #112240;border-top:3px solid {color};
                    border-radius:14px;padding:1.2rem 1rem;text-align:center;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
                        text-transform:uppercase;color:#4A6A8A;margin-bottom:0.4rem;">
                {label}
            </div>
            <div style="font-size:2rem;font-weight:800;color:{color};
                        font-family:'Space Mono',monospace;">
                {val}
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY BAR CHART (always visible)
# ─────────────────────────────────────────────────────────────────────────────
fig_acc = go.Figure()
acc_labels = ["Overall", "Glioma", "Meningioma", "No Tumor", "Pituitary"]
acc_values = [89, 92, 84, 95, 99]
acc_colors = ["#1CFFD4", "#FF4D6D", "#FFB347", "#00E5A0", "#5BA8FF"]

fig_acc.add_trace(go.Bar(
    x=acc_labels,
    y=acc_values,
    marker_color=acc_colors,
    text=[f"{v}%" for v in acc_values],
    textposition="outside",
    textfont=dict(color="#C8D8E8", size=13, family="Space Mono, monospace"),
    hovertemplate="<b>%{x}</b><br>Accuracy: %{y}%<extra></extra>",
))
fig_acc.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,22,40,0.6)",
    height=260,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis=dict(tickfont=dict(color="#6A8FAF", size=12), gridcolor="rgba(0,0,0,0)", zeroline=False, showline=False),
    yaxis=dict(range=[0, 115], ticksuffix="%", tickfont=dict(color="#4A6A8A", size=11),
               gridcolor="rgba(255,255,255,0.04)", zeroline=False, showline=False),
    showlegend=False,
    bargap=0.4,
)

st.markdown("""
<div style="background:#0A1628;border:1px solid #112240;border-radius:18px;padding:1.5rem 1.5rem 0.2rem;">
    <p style="font-size:0.7rem;font-weight:700;letter-spacing:0.15em;color:#3A5A7A;
              text-transform:uppercase;font-family:'Space Mono',monospace;margin:0 0 0.2rem;">
        ◈ Per-Class Accuracy Breakdown
    </p>
""", unsafe_allow_html=True)
st.plotly_chart(fig_acc, use_container_width=True, config={"displayModeBar": False})
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
section_label("Upload MRI Scan")
uploaded_file = st.file_uploader("upload", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    arr   = preprocess_input(np.array(image.resize((300, 300))).astype(np.float32))
    arr   = np.expand_dims(arr, axis=0)

    with st.spinner("Running inference…"):
        pred = model.predict(arr, verbose=0)

    probs   = pred[0]
    top_idx = int(np.argmax(probs))
    cls     = CLASS_NAMES[top_idx]
    conf    = float(probs[top_idx]) * 100
    info    = TUMOR_INFO[cls]
    color   = info["color"]
    glow    = info["glow"]

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Image  |  Result card ─────────────────────────────────────────────────
    img_col, res_col = st.columns([1, 1.7], gap="large")

    with img_col:
        section_label("Uploaded Scan")
        st.image(image, use_column_width=True)

    with res_col:
        st.markdown(f"""
        <div style="background:#080F1C;border:1.5px solid {color}44;border-radius:22px;
                    padding:2rem 2.2rem;box-shadow:0 0 60px {glow};margin-bottom:1rem;">

            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.4rem;">
                <div>
                    <div style="font-size:0.65rem;font-weight:700;letter-spacing:0.2em;color:#3A5A7A;
                                text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:0.35rem;">
                        Classification Result
                    </div>
                    <div style="font-size:2.4rem;font-weight:800;color:{color};line-height:1;letter-spacing:-0.02em;">
                        {info["label"]}
                    </div>
                </div>
                <span style="background:{color}18;color:{color};border:1px solid {color}44;
                             border-radius:99px;padding:0.3rem 1rem;font-size:0.68rem;
                             font-weight:700;letter-spacing:0.1em;font-family:'Space Mono',monospace;
                             white-space:nowrap;margin-top:0.3rem;">
                    {info["severity"]}
                </span>
            </div>

            <div style="margin-bottom:1.4rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.45rem;">
                    <span style="font-size:0.8rem;font-weight:600;color:#6A8FAF;">Confidence Score</span>
                    <span style="font-size:1rem;font-weight:800;color:{color};
                                 font-family:'Space Mono',monospace;">{conf:.2f}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:99px;height:8px;overflow:hidden;">
                    <div style="background:linear-gradient(90deg,{color}66,{color});
                                width:{conf:.1f}%;height:100%;border-radius:99px;
                                box-shadow:0 0 10px {color}55;"></div>
                </div>
            </div>

            <p style="color:#8AAAC0;font-size:0.88rem;line-height:1.7;margin-bottom:1.2rem;">
                {info["desc"]}
            </p>

            <div style="background:rgba(255,255,255,0.04);border-radius:10px;
                        padding:0.7rem 1rem;margin-bottom:1rem;display:flex;gap:0.7rem;align-items:flex-start;">
                <span style="color:{color};font-weight:700;font-size:0.78rem;white-space:nowrap;">Prognosis</span>
                <span style="color:#7A9AB8;font-size:0.82rem;line-height:1.5;">{info["prognosis"]}</span>
            </div>

            <div style="border-left:3px solid {color};padding:0.6rem 1rem;
                        border-radius:0 8px 8px 0;background:{color}0F;
                        color:#A0C0D8;font-size:0.83rem;line-height:1.5;">
                {info["note"]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability charts ────────────────────────────────────────────────────
    sorted_idx = np.argsort(probs)
    cls_s  = [CLASS_NAMES[i] for i in sorted_idx]
    pct_s  = [float(probs[i]) * 100 for i in sorted_idx]
    col_s  = [COLORS[c] if c == cls else COLORS[c] + "55" for c in cls_s]
    lbl_s  = [LABELS[c] for c in cls_s]
    txt_col= [COLORS[c] if c == cls else "#4A6A8A" for c in cls_s]

    # Horizontal bar
    fig_prob = go.Figure()
    fig_prob.add_trace(go.Bar(
        y=lbl_s, x=pct_s, orientation="h",
        marker_color=col_s,
        text=[f"  {p:.1f}%" for p in pct_s],
        textposition="outside",
        textfont=dict(color=txt_col, size=13, family="Space Mono, monospace"),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
    ))
    fig_prob.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=210, margin=dict(l=10, r=70, t=10, b=10),
        xaxis=dict(range=[0, 118], ticksuffix="%", tickfont=dict(color="#4A6A8A", size=11),
                   gridcolor="rgba(255,255,255,0.04)", zeroline=False, showline=False),
        yaxis=dict(tickfont=dict(color="#A0C0D8", size=13), gridcolor="rgba(0,0,0,0)",
                   zeroline=False, showline=False),
        showlegend=False, bargap=0.35,
    )

    # Donut
    fig_donut = go.Figure(data=[go.Pie(
        labels=[LABELS[c] for c in CLASS_NAMES],
        values=[float(probs[i]) * 100 for i in range(4)],
        hole=0.65,
        marker=dict(colors=[COLORS[c] for c in CLASS_NAMES], line=dict(color="#050C15", width=3)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
        sort=False,
    )])
    fig_donut.add_annotation(
        text=f"<b>{conf:.1f}%</b>", x=0.5, y=0.56,
        font=dict(size=20, color=color, family="Space Mono, monospace"),
        showarrow=False,
    )
    fig_donut.add_annotation(
        text="confidence", x=0.5, y=0.40,
        font=dict(size=11, color="#4A6A8A", family="DM Sans, sans-serif"),
        showarrow=False,
    )
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(font=dict(color="#7A9AB8", size=11), bgcolor="rgba(0,0,0,0)",
                    x=0.5, xanchor="center", y=-0.08, orientation="h"),
    )

    chart_l, chart_r = st.columns([1.6, 1], gap="large")
    with chart_l:
        st.markdown("""
        <div style="background:#0A1628;border:1px solid #112240;border-radius:18px;padding:1.5rem 1.5rem 0.2rem;">
            <p style="font-size:0.7rem;font-weight:700;letter-spacing:0.15em;color:#3A5A7A;
                      text-transform:uppercase;font-family:'Space Mono',monospace;margin:0 0 0.2rem;">
                ◈ All Class Probabilities
            </p>""", unsafe_allow_html=True)
        st.plotly_chart(fig_prob, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_r:
        st.markdown("""
        <div style="background:#0A1628;border:1px solid #112240;border-radius:18px;padding:1.5rem 1.5rem 0.2rem;">
            <p style="font-size:0.7rem;font-weight:700;letter-spacing:0.15em;color:#3A5A7A;
                      text-transform:uppercase;font-family:'Space Mono',monospace;margin:0 0 0.2rem;">
                ◈ Distribution
            </p>""", unsafe_allow_html=True)
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Symptoms + Treatment ──────────────────────────────────────────────────
    sym_col, treat_col = st.columns(2, gap="large")

    def detail_card(title, items, accent, bullet):
        rows = "".join(f"""
            <div style="display:flex;align-items:center;gap:0.75rem;
                        padding:0.6rem 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="color:{accent};font-weight:700;flex-shrink:0;">{bullet}</span>
                <span style="color:#A0C0D8;font-size:0.87rem;">{item}</span>
            </div>""" for item in items)
        return f"""
        <div style="background:#0A1628;border:1px solid #112240;border-top:3px solid {accent};
                    border-radius:18px;padding:1.5rem 1.6rem;height:100%;">
            <p style="font-size:0.7rem;font-weight:700;letter-spacing:0.15em;color:#3A5A7A;
                      text-transform:uppercase;font-family:'Space Mono',monospace;margin:0 0 0.8rem;">
                ◈ {title}
            </p>
            {rows}
        </div>"""

    with sym_col:
        st.markdown(detail_card("Common Symptoms", info["symptoms"], color, "▸"), unsafe_allow_html=True)
    with treat_col:
        st.markdown(detail_card("Treatment Options", info["treatment"], "#1CFFD4", "✓"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#0A1628;border:1px solid #112240;border-radius:14px;
                padding:1rem 1.5rem;display:flex;align-items:center;gap:1rem;">
        <span style="font-size:1.4rem;">⚕️</span>
        <p style="margin:0;font-size:0.82rem;color:#4A6A8A;line-height:1.5;">
            <strong style="color:#6A8FAF;">Medical Disclaimer:</strong>
            This tool is for educational and research purposes only and is
            <strong style="color:#6A8FAF;">not</strong> a substitute for professional medical advice,
            diagnosis, or treatment. Always consult a qualified healthcare provider.
        </p>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:3rem 2rem;">
        <div style="font-size:3.5rem;margin-bottom:1rem;">🔬</div>
        <p style="font-size:1.1rem;font-weight:600;color:#4A6A8A;">
            Upload a brain MRI scan above to begin classification
        </p>
        <p style="font-size:0.88rem;color:#2A4A6A;">Supported formats: JPG · PNG · JPEG</p>
        <div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;margin-top:2.5rem;">
            <span style="background:#FF4D6D11;border:1px solid #FF4D6D44;color:#FF4D6D;
                         padding:0.5rem 1.3rem;border-radius:99px;font-size:0.85rem;font-weight:600;">
                ⬤ Glioma · 92%
            </span>
            <span style="background:#FFB34711;border:1px solid #FFB34744;color:#FFB347;
                         padding:0.5rem 1.3rem;border-radius:99px;font-size:0.85rem;font-weight:600;">
                ⬤ Meningioma · 84%
            </span>
            <span style="background:#00E5A011;border:1px solid #00E5A044;color:#00E5A0;
                         padding:0.5rem 1.3rem;border-radius:99px;font-size:0.85rem;font-weight:600;">
                ⬤ No Tumor · 95%
            </span>
            <span style="background:#5BA8FF11;border:1px solid #5BA8FF44;color:#5BA8FF;
                         padding:0.5rem 1.3rem;border-radius:99px;font-size:0.85rem;font-weight:600;">
                ⬤ Pituitary · 99%
            </span>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
