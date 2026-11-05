import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from pathlib import Path
from tensorflow.keras.models import load_model

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

RESULTS_DIR = BASE_DIR / "results"
DATA_PATH = BASE_DIR / "dataset" / "processed" / "modis_cleaned.csv"

# ── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Forest Fire Detection",
    page_icon="🔥",
    layout="wide"
)

# ── Custom CSS — Dark wildfire theme ──────────────────
st.markdown("""
<style>
    /* Dark background */
    .stApp { background-color: #0f0f0f; color: #f0f0f0; }
    section[data-testid="stSidebar"] { background-color: #1a1a1a; border-right: 2px solid #ff4500; }

    /* Cards */
    .fire-card {
        background: linear-gradient(135deg, #1a1a1a, #2a1a0a);
        border: 1px solid #ff4500;
        border-radius: 16px;
        padding: 20px;
        margin: 10px 0;
    }
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #ff6600;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #0a2a0a, #1a3a1a);
        border: 2px solid #00cc44;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .result-medium {
        background: linear-gradient(135deg, #2a1a0a, #3a2a0a);
        border: 2px solid #ff8800;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .result-high {
        background: linear-gradient(135deg, #2a0a0a, #3a1a1a);
        border: 2px solid #ff0000;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .title-fire { color: #ff4500; font-size: 2rem; font-weight: bold; }
    .subtitle { color: #ff8c00; font-size: 1rem; }
    h1, h2, h3 { color: #ff6600 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #ff4500, #cc2200);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-size: 1rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #ff6600, #ff2200); }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='title-fire'>🔥 Forest Fire</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Detection System — Turkey 2020</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠  Accueil",
        "📊  Analyse du dataset",
        "🗺️  Carte des incendies",
        "🤖  Prédiction ANN",
        "🔁  Prédiction LSTM",
        "⚡  Prédiction ANN + LSTM",
    ])
    st.markdown("---")
    st.markdown("<small style='color:#888'>Données MODIS NASA 2020</small>", unsafe_allow_html=True)

# ── Chargement ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    ann = load_model(MODELS_DIR / "ann_model_final.h5")
    lstm = load_model(MODELS_DIR / "lstm_model_final.h5")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    enc_sat = joblib.load(MODELS_DIR / "encoder_satellite.pkl")
    enc_ins = joblib.load(MODELS_DIR / "encoder_instrument.pkl")
    enc_day = joblib.load(MODELS_DIR / "encoder_daynight.pkl")
    with open(MODELS_DIR / "feature_names.json") as f:
        features = json.load(f)
    return ann, lstm, scaler, enc_sat, enc_ins, enc_day, features

df = load_data()
ann_model, lstm_model, scaler, enc_sat, enc_ins, enc_day, features = load_models()

RISK_LABELS = {0: "🟢 Low Risk", 1: "🟠 Medium Risk", 2: "🔴 High Risk"}
RISK_COLORS = {0: "#00cc44", 1: "#ff8800", 2: "#ff0000"}
RISK_CSS    = {0: "result-low", 1: "result-medium", 2: "result-high"}

# ── Session state pour historique ─────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ══════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════
def gauge_chart(value, title="Risk Meter"):
    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")
    theta = np.linspace(np.pi, 0, 300)
    for i, (color, start, end) in enumerate([
        ("#00cc44", 0,   100),
        ("#ff8800", 100, 200),
        ("#ff0000", 200, 300),
    ]):
        ax.plot(np.cos(theta[start:end]), np.sin(theta[start:end]),
                color=color, linewidth=18, solid_capstyle="butt")
    angle = np.pi - (value / 100) * np.pi
    ax.annotate("", xy=(0.6 * np.cos(angle), 0.6 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.3, 1.2)
    ax.axis("off")
    ax.text(0, -0.2, f"{value:.0f}/100", ha="center", color="white",
            fontsize=16, fontweight="bold")
    ax.text(0, 1.1, title, ha="center", color="#ff6600", fontsize=12)
    ax.text(-1.2, -0.15, "LOW", color="#00cc44", fontsize=9)
    ax.text(-0.1, -0.15, "MED", color="#ff8800", fontsize=9)
    ax.text(0.85, -0.15, "HIGH", color="#ff0000", fontsize=9)
    plt.tight_layout()
    return fig

def prob_bar_chart(proba, title="Probability"):
    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0f0f0f")
    ax.set_facecolor("#1a1a1a")
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    colors = ["#00cc44", "#ff8800", "#ff0000"]
    bars = ax.barh(labels, [p * 100 for p in proba], color=colors,
                   height=0.5, edgecolor="#333")
    for bar, p in zip(bars, proba):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", color="white", fontsize=11)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Probability (%)", color="#aaa")
    ax.set_title(title, color="#ff6600", fontsize=12)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")
    plt.tight_layout()
    return fig

def history_chart():
    hist = st.session_state.history
    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0f0f0f")
    ax.set_facecolor("#1a1a1a")
    labels = [h["model"] + f" #{i+1}" for i, h in enumerate(hist)]
    values = [h["risk"] for h in hist]
    colors = [RISK_COLORS[v] for v in values]
    ax.bar(labels, values, color=colors, edgecolor="#333", width=0.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Low", "Medium", "High"], color="white")
    ax.set_title("Prediction History", color="#ff6600", fontsize=13)
    ax.tick_params(colors="white", axis="x", rotation=30)
    ax.spines[:].set_color("#333")
    plt.tight_layout()
    return fig

def prediction_form(key_prefix=""):
    col1, col2 = st.columns(2)
    with col1:
        latitude   = st.number_input("📍 Latitude",   value=37.0,  key=f"{key_prefix}_lat")
        longitude  = st.number_input("📍 Longitude",  value=35.0,  key=f"{key_prefix}_lon")
        brightness = st.number_input("🌡️ Brightness", value=320.0, key=f"{key_prefix}_bri")
        scan       = st.number_input("📡 Scan",        value=1.0,  key=f"{key_prefix}_sca")
        track      = st.number_input("📡 Track",       value=1.0,  key=f"{key_prefix}_tra")
        confidence = st.number_input("✅ Confidence",  value=80,   key=f"{key_prefix}_con")
    with col2:
        bright_t31 = st.number_input("🌡️ Bright_t31", value=290.0, key=f"{key_prefix}_b31")
        frp        = st.number_input("🔥 FRP",         value=10.0,  key=f"{key_prefix}_frp")
        daynight   = st.selectbox("🌙 Day/Night",   ["D", "N"],             key=f"{key_prefix}_dn")
        satellite  = st.selectbox("🛰️ Satellite",   ["Terra", "Aqua"],      key=f"{key_prefix}_sat")
        instrument = st.selectbox("🔬 Instrument",  ["MODIS"],              key=f"{key_prefix}_ins")
        version    = st.number_input("📋 Version",   value=6.1,             key=f"{key_prefix}_ver")

    from datetime import date
    today = date.today()
    input_dict = {
        "latitude": latitude, "longitude": longitude,
        "brightness": brightness, "scan": scan, "track": track,
        "satellite": enc_sat.transform([satellite])[0],
        "instrument": enc_ins.transform([instrument])[0],
        "confidence": confidence, "version": version,
        "bright_t31": bright_t31,
        "daynight": enc_day.transform([daynight])[0],
        "type": 0,
        "year": today.year, "month": today.month, "day": today.day,
        "hour": 12, "minute": 0
    }
    input_df = pd.DataFrame([input_dict])[features]
    input_scaled = scaler.transform(input_df)
    return input_scaled, latitude, longitude

def show_result(proba, pred, model_name, lat, lon):
    risk_score = int(pred * 50 + np.max(proba) * 30)
    risk_score = min(risk_score, 99)

    st.markdown(f"""
    <div class='{RISK_CSS[pred]}'>
        <h1 style='font-size:2.5rem; margin:0'>{RISK_LABELS[pred]}</h1>
        <p style='color:#aaa; margin:5px 0'>Model: {model_name} &nbsp;|&nbsp; Confidence: {np.max(proba)*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(prob_bar_chart(proba, f"{model_name} — Probabilities"))
    with col2:
        st.pyplot(gauge_chart(risk_score, "Risk Meter"))

    st.subheader("📍 Fire Location")
    map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
    st.map(map_df, zoom=6)

    if len(st.session_state.history) > 1:
        st.subheader("📈 Prediction History")
        st.pyplot(history_chart())

# ══════════════════════════════════════════════════════
# PAGE 1 — Accueil
# ══════════════════════════════════════════════════════
if page == "🏠  Accueil":
    st.markdown("<h1 class='title-fire'>🔥 Forest Fire Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Turkey 2020 — MODIS Satellite Data</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='fire-card'>
            <h3>📡 Data Source</h3>
            <p>NASA MODIS satellite — détection thermique des points de feu en Turquie 2020.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='fire-card'>
            <h3>🧠 AI Models</h3>
            <p>ANN — 88.4% accuracy<br>LSTM — 89.8% accuracy</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='fire-card'>
            <h3>🎯 Risk Levels</h3>
            <p>🟢 Low Risk — FRP &lt; 5<br>🟠 Medium Risk — FRP 5–15<br>🔴 High Risk — FRP ≥ 15</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PAGE 2 — Analyse
# ══════════════════════════════════════════════════════
if page == "📊  Analyse du dataset":
    st.title("📊 Analyse du dataset MODIS")
    st.dataframe(df.head(20), use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("🗂️ Lignes", df.shape[0])
    col2.metric("📋 Colonnes", df.shape[1])
    col3.metric("❌ Valeurs manquantes", df.isnull().sum().sum())
    st.subheader("📈 Statistiques générales")
    st.dataframe(df.describe(), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔗 Matrice de corrélation")
        st.image(str(RESULTS_DIR / "correlation_matrix.png"), use_container_width=True)
    with col2:
        st.subheader("🎯 Distribution des risques")
        st.image(str(RESULTS_DIR / "target_distribution.png"), use_container_width=True)

# ══════════════════════════════════════════════════════
# PAGE 3 — Carte
# ══════════════════════════════════════════════════════
if page == "🗺️  Carte des incendies":
    st.title("🗺️ Carte des points d'incendie — Turquie 2020")
    map_df = df[["latitude", "longitude"]].dropna().rename(columns={"latitude": "lat", "longitude": "lon"})
    st.metric("🔥 Points de feu détectés", len(map_df))
    st.map(map_df)

# ══════════════════════════════════════════════════════
# PAGE 4 — Prédiction ANN
# ══════════════════════════════════════════════════════
if page == "🤖  Prédiction ANN":
    st.title("🤖 Prédiction ANN")
    st.markdown("<div class='fire-card'>Entrez les valeurs MODIS pour prédire le niveau de risque d'incendie.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    input_scaled, lat, lon = prediction_form("ann")

    if st.button("🔥 Prédire avec ANN"):
        proba = ann_model.predict(input_scaled)[0]
        pred = int(np.argmax(proba))
        st.session_state.history.append({"model": "ANN", "risk": pred, "proba": proba.tolist()})
        st.markdown("---")
        st.subheader("🔥 Prediction Result")
        show_result(proba, pred, "ANN", lat, lon)

# ══════════════════════════════════════════════════════
# PAGE 5 — Prédiction LSTM
# ══════════════════════════════════════════════════════
if page == "🔁  Prédiction LSTM":
    st.title("🔁 Prédiction LSTM")
    st.markdown("<div class='fire-card'>Entrez les valeurs MODIS pour prédire le niveau de risque d'incendie.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    input_scaled, lat, lon = prediction_form("lstm")

    if st.button("🔥 Prédire avec LSTM"):
        input_lstm = input_scaled.reshape((1, 1, input_scaled.shape[1]))
        proba = lstm_model.predict(input_lstm)[0]
        pred = int(np.argmax(proba))
        st.session_state.history.append({"model": "LSTM", "risk": pred, "proba": proba.tolist()})
        st.markdown("---")
        st.subheader("🔥 Prediction Result")
        show_result(proba, pred, "LSTM", lat, lon)

# ══════════════════════════════════════════════════════
# PAGE 6 — ANN + LSTM ensemble
# ══════════════════════════════════════════════════════
if page == "⚡  Prédiction ANN + LSTM":
    st.title("⚡ Prédiction ANN + LSTM")
    st.markdown("<div class='fire-card'>Comparez les deux modèles en même temps sur les mêmes données.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    input_scaled, lat, lon = prediction_form("both")

    if st.button("🔥 Prédire avec ANN + LSTM"):
        proba_ann  = ann_model.predict(input_scaled)[0]
        input_lstm = input_scaled.reshape((1, 1, input_scaled.shape[1]))
        proba_lstm = lstm_model.predict(input_lstm)[0]

        pred_ann  = int(np.argmax(proba_ann))
        pred_lstm = int(np.argmax(proba_lstm))

        st.session_state.history.append({"model": "ANN",  "risk": pred_ann,  "proba": proba_ann.tolist()})
        st.session_state.history.append({"model": "LSTM", "risk": pred_lstm, "proba": proba_lstm.tolist()})

        st.markdown("---")
        st.subheader("🔥 Prediction Results")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""<div class='{RISK_CSS[pred_ann]}'>
                <h2>🤖 ANN</h2>
                <h1>{RISK_LABELS[pred_ann]}</h1>
                <p>Confidence: {np.max(proba_ann)*100:.1f}%</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class='{RISK_CSS[pred_lstm]}'>
                <h2>🔁 LSTM</h2>
                <h1>{RISK_LABELS[pred_lstm]}</h1>
                <p>Confidence: {np.max(proba_lstm)*100:.1f}%</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Probability Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(prob_bar_chart(proba_ann, "ANN — Probabilities"))
        with col2:
            st.pyplot(prob_bar_chart(proba_lstm, "LSTM — Probabilities"))

        st.subheader("🎯 Risk Meter Comparison")
        risk_ann  = int(pred_ann  * 50 + np.max(proba_ann)  * 30)
        risk_lstm = int(pred_lstm * 50 + np.max(proba_lstm) * 30)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(gauge_chart(min(risk_ann, 99),  "ANN Risk Meter"))
        with col2:
            st.pyplot(gauge_chart(min(risk_lstm, 99), "LSTM Risk Meter"))

        st.subheader("📍 Fire Location")
        map_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
        st.map(map_df, zoom=6)

        if len(st.session_state.history) > 1:
            st.subheader("📈 Prediction History")
            st.pyplot(history_chart())