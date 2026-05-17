import streamlit as st
import pandas as pd
import numpy as np
import json
import pydeck as pdk
import joblib
import os
from datetime import datetime

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Earthquake Early Warning App",
    layout="wide",
    page_icon="🌍"
)

# ------------------------------------------------------------
# THEME / VISUAL SETTINGS
# ------------------------------------------------------------
THEME_CONFIG_PATH = "onur/theme_config.json"

@st.cache_data
def load_theme_config(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

saved_theme = load_theme_config(THEME_CONFIG_PATH)

st.sidebar.header("🎨 Tema & Görsel Ayarlar")
selected_theme = st.sidebar.selectbox(
    "Tema", ["Aydınlık", "Koyu"], index=0 if saved_theme.get("theme") == "Aydınlık" else 1
)
accent_color = st.sidebar.color_picker(
    "Vurgu Rengi", saved_theme.get("accent_color", "#FF4B4B")
)

if selected_theme == "Koyu":
    bg_color = "#0E1117"
    card_color = "#161B22"
    text_color = "#E6EDF3"
else:
    bg_color = "#FFFFFF"
    card_color = "#F6F8FA"
    text_color = "#1F2328"

st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .metric-card {{
            background-color: {card_color};
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }}
        .accent {{
            color: {accent_color};
            font-weight: 700;
        }}
        .risk-bar > div > div {{
            background-color: {accent_color} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌍 EARLY WARNING EARTHQUAKE APP")
st.markdown(
    "Bu uygulama **deprem erken uyarı modeli**nin amaçlarını, hedef değişkenlerini ve tahminlerini **interaktif** şekilde sunar."
)

if st.sidebar.button("Tema Ayarlarını Kaydet"):
    try:
        os.makedirs(os.path.dirname(THEME_CONFIG_PATH), exist_ok=True)
        with open(THEME_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"theme": selected_theme, "accent_color": accent_color}, f, ensure_ascii=False, indent=4)
        st.sidebar.success("Tema ayarları kaydedildi.")
    except Exception as exc:
        st.sidebar.error(f"Tema kaydedilemedi: {exc}")

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
DATA_PATH = "data/earthquakes_featured.csv"
PRED_PATH = "onur/predictions_table_onur.csv"
META_PATH = "onur/OUTPUTS_onur.json"
MODEL_DAYS_PATH = "onur/model_deepfault_days.pkl"
MODEL_MAG_PATH = "onur/model_deepfault_mag.pkl"
RISK_LOG_PATH = "onur/risk_log.csv"

CORE_FEATURES = [
    "magnitude",
    "depth",
    "latitude",
    "longitude",
    "temperature",
    "humidity",
    "pressure",
    "moon_phase",
    "sunspot_number",
    "solar_flux",
    "b_value",
    "rolling_mean_magnitude_7d",
    "rolling_max_magnitude_7d",
    "rolling_mean_magnitude_30d",
    "rolling_max_magnitude_30d",
    "event_count_7d",
    "event_count_30d",
    "energy",
    "energy_release_rate",
    "energy_anomaly_score",
    "magnitude_trend",
    "magnitude_acceleration",
    "days_since_last_event",
    "days_since_last_major_event",
    "shallow_event_ratio",
    "deep_event_ratio",
    "magnitude_depth_ratio",
]

TARGETS_INFO = {
    "future_max_magnitude_7d": (
        "Önümüzdeki 7 gün içinde aynı grid/region için görülebilecek **maksimum deprem büyüklüğünü** tahmin eder."
    ),
    "days_to_next_major_event": (
        "Bir sonraki **büyük olay (>=4.0)** için tahmini gün sayısını üretir."
    ),
}

MAJOR_MAG = 4.0

# ------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------
@st.cache_data
def load_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_predictions():
    return pd.read_csv(PRED_PATH, low_memory=False)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH, low_memory=False)

@st.cache_resource
def load_models():
    model_mag = joblib.load(MODEL_MAG_PATH)
    model_days = joblib.load(MODEL_DAYS_PATH)
    return model_mag, model_days

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def format_delta(value):
    if pd.isna(value):
        return "-"
    return f"{value:+.2f}" if isinstance(value, (int, float, np.number)) else str(value)

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
metadata = None
pred_df = None
df = None
model_mag = None
model_days = None

try:
    metadata = load_metadata()
except Exception:
    metadata = None

try:
    pred_df = load_predictions()
except Exception:
    pred_df = None

try:
    df = load_dataset()
except Exception:
    df = None

try:
    model_mag, model_days = load_models()
except Exception:
    model_mag, model_days = None, None

# ------------------------------------------------------------
# USER-CONTROLLED THRESHOLDS & RISK WEIGHTS
# ------------------------------------------------------------
st.sidebar.header("🚨 Uyarı Eşikleri")
mag_threshold_high = st.sidebar.slider("Yüksek Risk Magnitude", 4.0, 7.0, 5.0, 0.1)
mag_threshold_mid = st.sidebar.slider("Orta Risk Magnitude", 3.0, mag_threshold_high, 4.0, 0.1)

short_days = st.sidebar.slider("Kısa Süre (gün)", 1, 21, 7)
mid_days = st.sidebar.slider("Orta Süre (gün)", short_days, 90, 30)

st.sidebar.header("📊 Risk Skoru Ağırlıkları")
w_mag = st.sidebar.slider("Magnitude Ağırlığı", 0.0, 1.0, 0.5, 0.05)
w_days = st.sidebar.slider("Gün Ağırlığı", 0.0, 1.0, 0.3, 0.05)
w_conf = st.sidebar.slider("Confidence Ağırlığı", 0.0, 1.0, 0.2, 0.05)

mag_scale = st.sidebar.slider("Magnitude Ölçek (Normalizasyon)", 5.0, 8.0, 7.0, 0.1)
days_scale = st.sidebar.slider("Gün Ölçek (Normalizasyon)", 7, 90, 30)

weight_sum = max(w_mag + w_days + w_conf, 1e-6)

# ------------------------------------------------------------
# PRESENTATION MODE
# ------------------------------------------------------------
pres_tab, analysis_tab, scenario_tab, map_tab = st.tabs(
    ["🎬 Sunum Modu", "📊 Analiz", "🧪 Senaryo", "🗺️ Harita"]
)

with pres_tab:
    st.subheader("🎯 Model Amacı ve Hedef Değişkenler")
    st.markdown(
        "Bu model, tarihsel deprem verilerini ve jeofiziksel sinyalleri kullanarak **gelecek 7 gün için maksimum büyüklük** ve **bir sonraki büyük depreme kalan gün** tahminlerini üretir."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Hedef 1: future_max_magnitude_7d**")
        st.info(TARGETS_INFO["future_max_magnitude_7d"])
    with col_b:
        st.markdown("**Hedef 2: days_to_next_major_event**")
        st.info(TARGETS_INFO["days_to_next_major_event"])

    st.markdown("### 🧭 Sunum Akışı")
    st.markdown(
        "1. Model amacı ve hedefler\n"
        "2. Veri ve model metrikleri\n"
        "3. Risk ve uyarı paneli\n"
        "4. Senaryo analizleri ve tahminler\n"
        "5. Harita tabanlı görselleştirme"
    )

    if metadata:
        st.markdown("### 📌 Model Meta Verileri")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model Versiyon", metadata.get("model_version", "-"))
        c2.metric("Veri Satır Sayısı", metadata.get("rows", "-"))
        c3.metric("Feature Sayısı", metadata.get("feature_count", "-"))
        c4.metric("Üretim Tarihi", metadata.get("generated_at", "-"))

        st.markdown("#### 📉 Cross-Validation Metikleri")
        st.json(metadata.get("cv_metrics", {}))
    else:
        st.warning("OUTPUTS_onur.json bulunamadı. Model meta verileri gösterilemiyor.")

with analysis_tab:
    # ------------------------------------------------------------
    # KPI PANEL
    # ------------------------------------------------------------
    st.subheader("📌 KPI Paneli (Trendli)")

    if pred_df is not None:
        tmp_pred = pred_df.copy()
        if "date" in tmp_pred.columns:
            tmp_pred["date"] = pd.to_datetime(tmp_pred["date"], errors="coerce")
            tmp_pred = tmp_pred.sort_values("date")

        latest = tmp_pred.tail(1).iloc[0]
        prev = tmp_pred.tail(2).head(1).iloc[0] if len(tmp_pred) > 1 else latest

        k1, k2, k3, k4 = st.columns(4)
        if "pred_future_max_magnitude_7d" in tmp_pred.columns:
            k1.metric(
                "Tahmini Max Magnitude",
                f"{latest['pred_future_max_magnitude_7d']:.2f}",
                delta=format_delta(latest["pred_future_max_magnitude_7d"] - prev["pred_future_max_magnitude_7d"])
            )
        if "pred_days_to_next_major_event" in tmp_pred.columns:
            k2.metric(
                "Sonraki Büyük Olay (gün)",
                f"{latest['pred_days_to_next_major_event']:.1f}",
                delta=format_delta(latest["pred_days_to_next_major_event"] - prev["pred_days_to_next_major_event"])
            )
        if "magnitude" in tmp_pred.columns:
            k3.metric(
                "Son Gözlenen Magnitude",
                f"{latest['magnitude']:.2f}",
                delta=format_delta(latest["magnitude"] - prev["magnitude"])
            )
        if "depth" in tmp_pred.columns:
            k4.metric(
                "Son Derinlik (km)",
                f"{latest['depth']:.1f}",
                delta=format_delta(latest["depth"] - prev["depth"])
            )
    else:
        st.info("KPI için predictions_table_onur.csv bulunamadı.")

    # ------------------------------------------------------------
    # EŞİK BAZLI UYARI PANELİ
    # ------------------------------------------------------------
    st.subheader("🚨 Eşik Bazlı Uyarı Paneli")
    risk_score = None
    risk_payload = {}

    if pred_df is not None:
        tmp_pred = pred_df.copy()
        if "date" in tmp_pred.columns:
            tmp_pred["date"] = pd.to_datetime(tmp_pred["date"]