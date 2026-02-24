import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import random
import time
from PIL import Image
import io
import base64
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Color Palette & Constants
# =============================================================================
BG_PRIMARY = "#060a14"
BG_CARD = "#0c1220"
BG_SECONDARY = "#131c2e"
BORDER = "#1e2b43"
ACCENT = "#2dd4bf"
TEXT_PRIMARY = "#e8ecf4"
TEXT_SECONDARY = "#c5cdd9"
TEXT_MUTED = "#7a8599"

TUMOR_INFO = {
    "glioma": {
        "label": "Glioma",
        "color": "#ef4444",
        "severity": "High Concern",
        "severity_level": "high",
        "description": "Gliomas originate from glial cells in the brain or spine and are among the most common primary brain tumors. They can be aggressive and may require immediate treatment.",
        "symptoms": ["Persistent headaches", "Seizures", "Vision problems", "Speech difficulty", "Cognitive changes"],
        "treatment": ["Surgical resection", "Radiation therapy", "Chemotherapy (Temozolomide)", "Targeted therapy"],
        "prevalence": "~33% of all brain tumors",
        "survival_rate": "5-year survival varies by grade",
    },
    "meningioma": {
        "label": "Meningioma",
        "color": "#f59e0b",
        "severity": "Moderate Concern",
        "severity_level": "moderate",
        "description": "Meningiomas arise from the meninges, the membranes surrounding the brain and spinal cord. They are often benign and slow-growing, but may require monitoring or treatment.",
        "symptoms": ["Gradual headaches", "Memory issues", "Hearing loss", "Visual disturbances", "Weakness in limbs"],
        "treatment": ["Active monitoring", "Surgical removal", "Stereotactic radiosurgery", "Radiation therapy"],
        "prevalence": "~30% of all brain tumors",
        "survival_rate": "~80% five-year survival",
    },
    "notumor": {
        "label": "No Tumor",
        "color": "#22c55e",
        "severity": "All Clear",
        "severity_level": "clear",
        "description": "No tumor detected in the MRI scan. The brain structure appears normal based on the AI classification model. Regular check-ups are still recommended.",
        "symptoms": ["No concerning indicators detected"],
        "treatment": ["Routine neurological check-ups", "Healthy lifestyle maintenance"],
        "prevalence": "N/A",
        "survival_rate": "N/A",
    },
    "pituitary": {
        "label": "Pituitary",
        "color": "#38bdf8",
        "severity": "Requires Attention",
        "severity_level": "moderate",
        "description": "Pituitary tumors occur in the pituitary gland at the base of the brain. They often affect hormone production and are usually treatable with good outcomes.",
        "symptoms": ["Hormonal imbalance", "Vision disturbance", "Chronic headaches", "Fatigue", "Unexplained weight changes"],
        "treatment": ["Hormone therapy", "Transsphenoidal surgery", "Medication (Cabergoline)", "Radiation therapy"],
        "prevalence": "~17% of all brain tumors",
        "survival_rate": "~95% five-year survival",
    },
}

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# =============================================================================
# Styles
# =============================================================================
def inject_styles():
    st.markdown(f"""
    <style>
        /* ---- Global ---- */
        .stApp {{
            background-color: {BG_PRIMARY};
        }}
        section[data-testid="stSidebar"] {{
            background-color: {BG_CARD};
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] * {{
            color: {TEXT_SECONDARY} !important;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: {TEXT_PRIMARY} !important;
        }}

        /* ---- Hide default header/footer ---- */
        header[data-testid="stHeader"] {{
            background: transparent;
        }}
        footer {{
            visibility: hidden;
        }}

        /* ---- Typography ---- */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
            color: {TEXT_PRIMARY} !important;
        }}
        p, li, span, div, label {{
            color: {TEXT_SECONDARY};
        }}

        /* ---- File uploader fixes ---- */
        [data-testid="stFileUploader"] {{
            background-color: transparent !important;
        }}
        [data-testid="stFileUploader"] section {{
            background-color: {BG_SECONDARY} !important;
            border: 2px dashed {BORDER} !important;
            border-radius: 16px !important;
            padding: 40px 20px !important;
            transition: border-color 0.2s ease;
        }}
        [data-testid="stFileUploader"] section:hover {{
            border-color: {ACCENT} !important;
        }}
        [data-testid="stFileUploader"] section * {{
            color: {TEXT_SECONDARY} !important;
        }}
        [data-testid="stFileUploader"] small {{
            color: {TEXT_MUTED} !important;
        }}
        [data-testid="stFileUploader"] button {{
            background-color: {ACCENT} !important;
            color: {BG_PRIMARY} !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }}
        [data-testid="stFileUploaderDropzoneInstructions] span {{
            color: {TEXT_SECONDARY} !important;
        }}

        /* Uploaded file name */
        [data-testid="stFileUploaderFile"] {{
            background-color: {BG_SECONDARY} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 8px !important;
        }}
        [data-testid="stFileUploaderFile"] * {{
            color: {TEXT_SECONDARY} !important;
        }}

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background-color: {BG_SECONDARY};
            border-radius: 10px;
            padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            background-color: transparent;
            border: none;
            border-radius: 8px;
            color: {TEXT_MUTED} !important;
            padding: 8px 16px;
            font-size: 14px;
        }}
        .stTabs [aria-selected="true"] {{
            background-color: {BG_CARD} !important;
            color: {ACCENT} !important;
            border: 1px solid {BORDER} !important;
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            padding-top: 16px;
        }}

        /* ---- Metric cards ---- */
        [data-testid="stMetric"] {{
            background-color: {BG_SECONDARY};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 16px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {TEXT_MUTED} !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {TEXT_PRIMARY} !important;
        }}

        /* ---- Expander ---- */
        .streamlit-expanderHeader {{
            background-color: {BG_SECONDARY} !important;
            border: 1px solid {BORDER} !important;
            border-radius: 12px !important;
            color: {TEXT_PRIMARY} !important;
        }}
        .streamlit-expanderContent {{
            background-color: {BG_CARD} !important;
            border: 1px solid {BORDER} !important;
        }}

        /* ---- Progress bars ---- */
        .stProgress > div > div > div > div {{
            border-radius: 8px;
        }}

        /* ---- Custom card class ---- */
        .card {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 24px;
        }}
        .card-inner {{
            background-color: {BG_SECONDARY};
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 16px;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .feature-card {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 20px;
            text-align: center;
            transition: border-color 0.2s ease;
        }}
        .feature-card:hover {{
            border-color: rgba(45,212,191,0.25);
        }}
        .step-num {{
            font-family: monospace;
            font-size: 11px;
            color: rgba(45,212,191,0.55);
            font-weight: 700;
        }}
        .symptom-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            background-color: {BG_SECONDARY};
            border-radius: 10px;
            padding: 12px 16px;
            margin-bottom: 8px;
        }}
        .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# Prediction logic (simulated)
# =============================================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tumor_model.keras")

model = load_model()

def real_prediction(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((300, 300))

    img_array = np.array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)[0]

    probs = {
        CLASS_NAMES[i]: round(float(preds[i] * 100), 2)
        for i in range(len(CLASS_NAMES))
    }

    top_class = CLASS_NAMES[np.argmax(preds)]
    confidence = probs[top_class]

    return {
        "class_name": top_class,
        "confidence": confidence,
        "probabilities": probs,
    }
# =============================================================================
# Chart helpers
# =============================================================================



def create_donut_chart(probs):
    labels = [TUMOR_INFO[k]["label"] for k in probs]
    values = list(probs.values())
    colors = [TUMOR_INFO[k]["color"] for k in probs]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0c1220", width=3)),
        textinfo="percent",
        textposition="inside",
        textfont=dict(color=TEXT_PRIMARY, size=13),
        insidetextorientation="horizontal",
        hovertemplate="<b>%{label}</b><br>%{value:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SECONDARY),
        showlegend=True,
        legend=dict(
            font=dict(color=TEXT_SECONDARY, size=12),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=10, r=10, t=10, b=40),
        height=280,
    )
    return fig


def create_gauge(value, color, label):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix="%", font=dict(color=color, size=36)),
        title=dict(text=label, font=dict(color=TEXT_SECONDARY, size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(color=TEXT_MUTED, size=11), dtick=25),
            bar=dict(color=color, thickness=0.7),
            bgcolor=BG_SECONDARY,
            bordercolor=BORDER,
            borderwidth=1,
            steps=[
                dict(range=[0, 40], color=f"{BORDER}"),
                dict(range=[40, 70], color=f"{BG_SECONDARY}"),
                dict(range=[70, 100], color=f"{BORDER}"),
            ],
            threshold=dict(line=dict(color=color, width=3), thickness=0.8, value=value),
        ),
    ))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SECONDARY),
        margin=dict(l=30, r=30, t=30, b=10),
        height=220,
    )
    return fig


def create_radar_chart(probs):
    labels = [TUMOR_INFO[k]["label"] for k in probs]
    values = list(probs.values())
    colors_list = [TUMOR_INFO[k]["color"] for k in probs]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(45,212,191,0.1)",
        line=dict(color=ACCENT, width=2),
        marker=dict(color=colors_list + [colors_list[0]], size=8),
        name="Probability",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=BORDER,
                tickfont=dict(color=TEXT_MUTED, size=10),
                ticksuffix="%",
            ),
            angularaxis=dict(
                gridcolor=BORDER,
                tickfont=dict(color=TEXT_SECONDARY, size=12),
            ),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SECONDARY),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=280,
    )
    return fig


# =============================================================================
# Sidebar
# =============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 16px;">
            <div style="font-size: 32px; margin-bottom: 8px;">🧠</div>
            <h2 style="margin: 0; font-size: 18px; color: {TEXT_PRIMARY} !important;">MRI Classifier</h2>
            <p style="font-size: 12px; color: {TEXT_MUTED}; margin-top: 4px;">v2.0 &bull; EfficientNet-B3</p>
        </div>
        <hr style="border-color: {BORDER}; margin: 16px 0;">
        """, unsafe_allow_html=True)

        st.markdown(f"#### Model Specifications")
        specs = {
            "Architecture": "EfficientNet-B3",
            "Input Size": "300 x 300 px",
            "Parameters": "~5.3M",
            "Training Acc.": "87.5%",
            "Framework": "Tensorflow",
        }
        for k, v in specs.items():
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid {BORDER};">
                <span style="color: {TEXT_MUTED}; font-size: 13px;">{k}</span>
                <span style="color: {TEXT_PRIMARY}; font-size: 13px; font-weight: 600;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"#### Detectable Classes")
        for cls in CLASS_NAMES:
            info = TUMOR_INFO[cls]
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; padding: 8px 12px; margin-bottom: 6px;
                        background-color: {BG_SECONDARY}; border-radius: 10px; border: 1px solid {BORDER};">
                <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {info['color']};"></div>
                <span style="color: {TEXT_SECONDARY}; font-size: 13px; font-weight: 500;">{info['label']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <br>
        <div style="background-color: {BG_SECONDARY}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px;">
            <p style="color: {TEXT_MUTED}; font-size: 11px; text-align: center; margin: 0;">
                For research and educational purposes only. Not a substitute for professional medical diagnosis.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# Hero Section
# =============================================================================
def render_hero():
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0 10px;">
        <div style="display: inline-flex; align-items: center; gap: 8px;
                    border: 1px solid rgba(45,212,191,0.2); background-color: rgba(45,212,191,0.05);
                    border-radius: 999px; padding: 6px 16px; margin-bottom: 16px;">
            <span style="font-size: 14px;">🧠</span>
            <span style="color: {ACCENT}; font-size: 12px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;">
                AI-Powered Diagnostics
            </span>
        </div>
        <h1 style="font-size: 42px; font-weight: 800; margin: 0; color: {TEXT_PRIMARY} !important;">
            Brain Tumor MRI <span style="color: {ACCENT};">Classifier</span>
        </h1>
        <p style="color: {TEXT_MUTED}; font-size: 16px; max-width: 560px; margin: 12px auto 0; line-height: 1.6;">
            Upload an MRI scan and receive instant AI-powered classification
            across 4 tumor types with detailed probability analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Feature cards
# =============================================================================
def render_features():
    features = [
        ("⚡", "Real-time Analysis", "Instant classification results powered by deep learning"),
        ("🧠", "EfficientNet Model", "State-of-the-art CNN architecture for accurate predictions"),
        ("📊", "Detailed Reports", "Comprehensive probability breakdown and clinical insights"),
        ("🔒", "Private & Secure", "All processing happens locally \u2014 your data never leaves"),
    ]
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i]:
            st.markdown(f"""
            <div class="feature-card">
                <div style="background-color: rgba(45,212,191,0.09); width: 40px; height: 40px; border-radius: 10px;
                            display: flex; align-items: center; justify-content: center; margin: 0 auto 12px; font-size: 18px;">
                    {icon}
                </div>
                <h4 style="font-size: 13px; font-weight: 700; color: {TEXT_PRIMARY} !important; margin: 0 0 6px;">{title}</h4>
                <p style="font-size: 11px; color: {TEXT_MUTED}; margin: 0; line-height: 1.5;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# How it works
# =============================================================================
def render_how_it_works():
    st.markdown(f"""
    <div style="text-align: center; margin: 40px 0 20px;">
        <h2 style="font-size: 22px; font-weight: 700; color: {TEXT_PRIMARY} !important; margin: 0;">How It Works</h2>
        <p style="color: {TEXT_MUTED}; font-size: 14px; margin-top: 6px;">Simple four-step process from upload to diagnosis</p>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("01", "📤", "Upload Scan", "Drop or select an MRI brain scan image in JPG or PNG format"),
        ("02", "🔬", "AI Processing", "EfficientNet analyzes the scan through multiple neural layers"),
        ("03", "📋", "Classification", "The model classifies the scan into one of 4 tumor categories"),
        ("04", "📈", "Results", "View detailed probabilities, charts, and clinical information"),
    ]
    cols = st.columns(4)
    for i, (num, icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="feature-card" style="text-align: left;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                    <div style="background-color: rgba(45,212,191,0.09); width: 36px; height: 36px; border-radius: 10px;
                                display: flex; align-items: center; justify-content: center; font-size: 16px;">
                        {icon}
                    </div>
                    <span class="step-num">{num}</span>
                </div>
                <h4 style="font-size: 14px; font-weight: 600; color: {TEXT_PRIMARY} !important; margin: 0 0 6px;">{title}</h4>
                <p style="font-size: 12px; color: {TEXT_MUTED}; margin: 0; line-height: 1.5;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    # Class badges
    st.markdown(f"""
    <div style="text-align: center; margin-top: 24px;">
        <p style="font-size: 11px; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; margin-bottom: 12px;">Detectable Tumor Classes</p>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
    """, unsafe_allow_html=True)
    badges_html = ""
    for cls in CLASS_NAMES:
        info = TUMOR_INFO[cls]
        badges_html += f"""
        <div style="display: inline-flex; align-items: center; gap: 8px; border: 1px solid {info['color']};
                    border-radius: 999px; padding: 6px 14px; opacity: 0.85;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background-color: {info['color']};"></div>
            <span style="font-size: 12px; font-weight: 600; color: {info['color']};">{info['label']}</span>
        </div>
        """
    st.markdown(f"""
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            {badges_html}
        </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Results
# =============================================================================
def render_results(result, image_bytes):
    cls = result["class_name"]
    info = TUMOR_INFO[cls]
    probs = result["probabilities"]

    st.markdown(f"""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
        <h2 style="font-size: 22px; font-weight: 700; color: {TEXT_PRIMARY} !important; margin: 0;">Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)

    # ---------- Top row: Image / Diagnosis / Gauge ----------
    col_img, col_diag, col_gauge = st.columns([1, 1.2, 1])

    with col_img:
        st.markdown(f"""
        <div class="card" style="padding: 16px;">
            <div style="background-color: rgba(0,0,0,0.4); border-radius: 12px; overflow: hidden; position: relative; text-align: center;">
                <img src="data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                     style="max-height: 280px; border-radius: 12px; object-fit: contain; width: 100%;" />
                <div style="position: absolute; top: 12px; right: 12px;
                            background-color: rgba(0,0,0,0.5); color: {info['color']};
                            border: 1px solid {info['color']}; border-radius: 999px;
                            padding: 4px 12px; font-size: 11px; font-weight: 700;">
                    Analyzed
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_diag:
        severity_icon = "🔴" if info["severity_level"] == "high" else ("🟡" if info["severity_level"] == "moderate" else "🟢")
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        st.markdown(f"""
        <div class="card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
                <span style="font-size: 18px;">🧠</span>
                <h3 style="margin: 0; font-size: 16px; color: {TEXT_PRIMARY} !important;">Diagnosis</h3>
            </div>
            <h2 style="font-size: 26px; font-weight: 800; color: {info['color']}; margin: 0 0 8px;">{info['label']}</h2>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <span>{severity_icon}</span>
                <span class="badge" style="border: 1px solid {info['color']}; color: {info['color']}; font-size: 11px;">
                    {info['severity']}
                </span>
            </div>
            <p style="font-size: 13px; color: {TEXT_MUTED}; line-height: 1.6; margin-bottom: 16px;">
                {info['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)

        for k, v in sorted_probs:
            p_info = TUMOR_INFO[k]
            st.markdown(f"""
            <div style="margin-bottom: 8px; padding: 0 24px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-size: 12px; color: {TEXT_MUTED};">{p_info['label']}</span>
                    <span style="font-size: 12px; font-family: monospace; color: {p_info['color']}; font-weight: 600;">{v:.1f}%</span>
                </div>
                <div style="width: 100%; height: 6px; background-color: {BG_SECONDARY}; border-radius: 4px; overflow: hidden;">
                    <div style="width: {v}%; height: 100%; background-color: {p_info['color']}; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_gauge:
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="font-size: 18px;">📈</span>
                <h3 style="margin: 0; font-size: 16px; color: {TEXT_PRIMARY} !important;">Confidence</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_gauge(result["confidence"], info["color"], info["label"]),
                        use_container_width=True, config={"displayModeBar": False})

        st.markdown(f"""
        <div style="display: flex; gap: 12px; margin-top: -10px;">
            <div class="card-inner" style="flex: 1; text-align: center;">
                <p style="font-size: 11px; color: {TEXT_MUTED}; margin: 0;">Model</p>
                <p style="font-size: 13px; color: {TEXT_PRIMARY}; font-weight: 700; margin: 4px 0 0;">EfficientNet</p>
            </div>
            <div class="card-inner" style="flex: 1; text-align: center;">
                <p style="font-size: 11px; color: {TEXT_MUTED}; margin: 0;">Classes</p>
                <p style="font-size: 13px; color: {TEXT_PRIMARY}; font-weight: 700; margin: 4px 0 0;">4 Types</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- Charts row ----------
   
    # ---------- Radar chart ----------
    st.markdown("<br>", unsafe_allow_html=True)
    col_radar, col_donut = st.columns(2)

    with col_radar:
        st.markdown(f"""
        <div class="card" style="padding: 20px 20px 8px;">
            <h3 style="font-size: 15px; font-weight: 600; color: {TEXT_PRIMARY} !important; margin: 0 0 12px;">Radar Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_radar_chart(probs), use_container_width=True, config={"displayModeBar": False})

    with col_donut:
        st.markdown(f"""
        <div class="card" style="padding: 20px 20px 8px;">
            <h3 style="font-size: 15px; font-weight: 600; color: {TEXT_PRIMARY} !important; margin: 0 0 12px;">Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_donut_chart(probs), use_container_width=True, config={"displayModeBar": False})

    # ---------- Clinical tabs ----------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card" style="padding: 24px 28px 12px;">
        <h3 style="font-size: 16px; font-weight: 600; color: {TEXT_PRIMARY} !important; margin: 0 0 4px;">Clinical Information</h3>
    </div>
    """, unsafe_allow_html=True)

    tab_symptoms, tab_treatment, tab_stats = st.tabs(["🩺 Symptoms", "💊 Treatment", "📊 Statistics"])

    with tab_symptoms:
        cols = st.columns(2)
        for i, symptom in enumerate(info["symptoms"]):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="symptom-item">
                    <div class="dot" style="background-color: {info['color']};"></div>
                    <span style="font-size: 14px; color: {TEXT_SECONDARY};">{symptom}</span>
                </div>
                """, unsafe_allow_html=True)

    with tab_treatment:
        cols = st.columns(2)
        for i, treat in enumerate(info["treatment"]):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="symptom-item">
                    <div style="width: 22px; height: 22px; border-radius: 50%; background-color: rgba(45,212,191,0.13);
                                display: flex; align-items: center; justify-content: center; flex-shrink: 0;
                                color: {ACCENT}; font-size: 11px; font-weight: 700;">{i+1}</div>
                    <span style="font-size: 14px; color: {TEXT_SECONDARY};">{treat}</span>
                </div>
                """, unsafe_allow_html=True)

    with tab_stats:
        col_prev, col_surv = st.columns(2)
        with col_prev:
            st.markdown(f"""
            <div class="card-inner">
                <p style="font-size: 11px; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 1.5px; margin: 0;">Prevalence</p>
                <p style="font-size: 20px; font-weight: 700; color: {TEXT_PRIMARY}; margin: 8px 0 0;">{info['prevalence']}</p>
            </div>
            """, unsafe_allow_html=True)
        with col_surv:
            st.markdown(f"""
            <div class="card-inner">
                <p style="font-size: 11px; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 1.5px; margin: 0;">Survival Rate</p>
                <p style="font-size: 20px; font-weight: 700; color: {TEXT_PRIMARY}; margin: 8px 0 0;">{info['survival_rate']}</p>
            </div>
            """, unsafe_allow_html=True)

    # ---------- Disclaimer ----------
    st.markdown(f"""
    <div style="background-color: {BG_SECONDARY}; border: 1px solid {BORDER}; border-radius: 12px;
                padding: 14px 20px; margin-top: 24px; text-align: center;">
        <p style="font-size: 12px; color: {TEXT_MUTED}; margin: 0; line-height: 1.6;">
            This tool is for educational and research purposes only. It is not a substitute for
            professional medical diagnosis. Always consult a qualified healthcare provider for medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Main
# =============================================================================
def main():
    inject_styles()
    render_sidebar()
    render_hero()

    st.markdown("<br>", unsafe_allow_html=True)
    render_features()
    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- Upload ----------
    uploaded_file = st.file_uploader(
        "Upload MRI Scan",
        type=["jpg", "jpeg", "png"],
        help="Drop or select an MRI brain scan image",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()

        # Run prediction (simulate)
        if "prediction" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
            with st.spinner(""):
                # Custom spinner
                st.markdown(f"""
                <div style="display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 20px;">
                    <div style="display: flex; align-items: center; gap: 12px; background-color: {BG_CARD};
                                border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 24px;">
                        <div style="width: 20px; height: 20px; border: 2px solid {ACCENT}; border-top-color: transparent;
                                    border-radius: 50%; animation: spin 0.8s linear infinite;"></div>
                        <div>
                            <p style="color: {TEXT_PRIMARY}; font-size: 14px; font-weight: 600; margin: 0;">Analyzing MRI Scan...</p>
                            <p style="color: {TEXT_MUTED}; font-size: 12px; margin: 2px 0 0;">Running EfficientNet classification model</p>
                        </div>
                    </div>
                    <style>@keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}</style>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(1.5)

            st.session_state["prediction"] = real_prediction(image_bytes)
            st.session_state["last_file"] = uploaded_file.name
            st.rerun()

        render_results(st.session_state["prediction"], image_bytes)

    else:
        st.session_state.pop("prediction", None)
        st.session_state.pop("last_file", None)
        render_how_it_works()

    # ---------- Footer ----------
    st.markdown(f"""
    <div style="border-top: 1px solid {BORDER}; margin-top: 40px; padding-top: 20px; text-align: center;">
        <p style="font-size: 12px; color: {TEXT_MUTED};">
            Brain Tumor MRI Classifier &bull; Powered by EfficientNet &bull; For research and educational purposes only
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
