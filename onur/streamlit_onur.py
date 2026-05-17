import streamlit as st
import pandas as pd
import numpy as np
import json
import pydeck as pdk
import joblib
from datetime import datetime

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Earthquake Early Warning App",
    layout="wide",
    page_icon="🌍"
)

st.title("🌍 EARLY WARNING EARTHQUAKE APP")
st.markdown(
    "Bu uygulama **deprem erken uyarı modeli**nin amaçlarını, hedef değişkenlerini ve tahminlerini **interaktif** şekilde sunar."
)

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
DATA_PATH = "data/earthquakes_featured.csv"
PRED_PATH = "onur/predictions_table_onur.csv"
META_PATH = "onur/OUTPUTS_onur.json"
MODEL_DAYS_PATH = "onur/model_deepfault_days.pkl"
MODEL_MAG_PATH = "onur/model_deepfault_mag.pkl"

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
# OVERVIEW
# ------------------------------------------------------------
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

if metadata:
    st.markdown("### 📌 Model Meta Verileri")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Versiyon", metadata.get("model_version", "-"))
    c2.metric("Veri Satır Sayısı", metadata.get("rows", "-"))
    c3.metric("Feature Sayısı", metadata.get("feature_count", "-"))
    c4.metric("Üretim Tarihi", metadata.get("generated_at", "-"))

    st.markdown("#### ✅ En iyi hiperparametreler")
    st.json(metadata.get("best_params", {}))

    st.markdown("#### 📉 Cross-Validation Metikleri")
    st.json(metadata.get("cv_metrics", {}))
else:
    st.warning("OUTPUTS_onur.json bulunamadı. Model meta verileri gösterilemiyor.")

# ------------------------------------------------------------
# DATA SUMMARY
# ------------------------------------------------------------
st.subheader("📊 Veri Özeti")
if df is not None:
    if "time" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    col1, col2, col3 = st.columns(3)
    col1.metric("Toplam Kayıt", f"{df.shape[0]:,}")
    col2.metric("Kolon Sayısı", f"{df.shape[1]:,}")
    if "date" in df.columns:
        col3.metric("Tarih Aralığı", f"{df['date'].min().date()} → {df['date'].max().date()}")

    with st.expander("Örnek Veri"):
        st.dataframe(df.head(50))

    if "magnitude" in df.columns:
        st.markdown("**Büyüklük Dağılımı**")
        st.bar_chart(df["magnitude"].value_counts().sort_index())

    if "region" in df.columns:
        st.markdown("**Bölge Dağılımı (Top 20)**")
        region_counts = df["region"].value_counts().head(20)
        st.bar_chart(region_counts)
else:
    st.warning("data/earthquakes_featured.csv bulunamadı veya okunamadı.")

# ------------------------------------------------------------
# INTERACTIVE WHAT-IF PREDICTION
# ------------------------------------------------------------
st.subheader("🧪 Senaryo Analizi (What-if)")

if pred_df is None and df is None:
    st.error("Tahmin veya ana veri seti yok. Önce veri dosyalarını yüklemelisin.")
else:
    base_df = pred_df if pred_df is not None else df

    if "date" in base_df.columns:
        base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce")

    # Sidebar controls
    st.sidebar.header("🧭 Senaryo Seçimi")

    if "region" in base_df.columns:
        region_list = sorted(base_df["region"].dropna().unique().tolist())
        selected_region = st.sidebar.selectbox("Bölge", ["Tümü"] + region_list)
    else:
        selected_region = "Tümü"

    if "date" in base_df.columns:
        min_date = base_df["date"].min().date()
        max_date = base_df["date"].max().date()
        selected_date = st.sidebar.date_input("Tarih", value=max_date, min_value=min_date, max_value=max_date)
    else:
        selected_date = None

    filtered = base_df.copy()
    if selected_region != "Tümü" and "region" in filtered.columns:
        filtered = filtered[filtered["region"] == selected_region]

    if selected_date and "date" in filtered.columns:
        filtered = filtered[filtered["date"].dt.date <= selected_date]

    if len(filtered) == 0:
        st.warning("Seçilen filtrelerle veri bulunamadı. Filtreleri genişlet.")
    else:
        base_row = filtered.tail(1).iloc[0].to_dict()

        st.markdown("**Seçilen baz kayıt**")
        st.dataframe(pd.DataFrame([base_row]))

        # Editable feature panel
        st.markdown("### 🔧 Özellikleri Değiştir")
        updated_row = base_row.copy()

        editable_features = [f for f in CORE_FEATURES if f in base_row]
        if not editable_features:
            st.info("Düzenlenebilir feature bulunamadı.")
        else:
            cols = st.columns(3)
            for idx, feature in enumerate(editable_features):
                col = cols[idx % 3]
                value = base_row.get(feature, 0.0)

                if isinstance(value, (int, float, np.number)):
                    if df is not None and feature in df.columns:
                        vmin = float(np.nanpercentile(df[feature], 5))
                        vmax = float(np.nanpercentile(df[feature], 95))
                    else:
                        vmin = float(value) - abs(float(value))
                        vmax = float(value) + abs(float(value)) + 1

                    if vmin == vmax:
                        vmin, vmax = vmin - 1, vmax + 1

                    updated_row[feature] = col.slider(
                        feature,
                        min_value=vmin,
                        max_value=vmax,
                        value=float(value),
                        step=(vmax - vmin) / 100 if vmax != vmin else 0.1,
                    )
                else:
                    updated_row[feature] = col.text_input(feature, value=str(value))

        # Recompute date features if user chooses
        if "date" in updated_row and selected_date:
            date_obj = pd.to_datetime(selected_date)
            updated_row["date"] = date_obj
            updated_row["year"] = date_obj.year
            updated_row["month"] = date_obj.month
            updated_row["day"] = date_obj.day
            updated_row["day_of_week"] = date_obj.dayofweek
            updated_row["season"] = date_obj.month % 12 // 3 + 1

        # Prediction
        st.markdown("### ✅ Tahmin Sonuçları")
        if metadata and model_mag and model_days:
            feature_list = metadata.get("features", [])
            X_row = pd.DataFrame([updated_row])

            # Ensure all features exist
            for feat in feature_list:
                if feat not in X_row.columns:
                    X_row[feat] = 0

            X_row = X_row[feature_list]

            # Predict
            pred_mag = float(model_mag.predict(X_row)[0])
            pred_days = float(model_days.predict(X_row)[0])

            # Confidence (tree variance)
            mag_tree_preds = np.array([tree.predict(X_row) for tree in model_mag.estimators_]).reshape(-1)
            days_tree_preds = np.array([tree.predict(X_row) for tree in model_days.estimators_]).reshape(-1)

            std_mag = float(np.std(mag_tree_preds))
            std_days = float(np.std(days_tree_preds))

            cv_mag = std_mag / pred_mag if pred_mag > 0 else np.inf
            cv_days = std_days / pred_days if pred_days > 0 else np.inf

            confidence_mag = 1 / (1 + cv_mag)
            confidence_days = 1 / (1 + cv_days)
            confidence = float((confidence_mag + confidence_days) / 2)

            col1, col2, col3 = st.columns(3)
            col1.metric("Tahmini Max Magnitude (7g)", f"{pred_mag:.2f}")
            col2.metric("Sonraki Büyük Olay (gün)", f"{pred_days:.1f}")
            col3.metric("Confidence Skoru", f"{confidence:.2f}")

            st.markdown("**Not:** Büyük olay eşiği = 4.0. Tahmin edilen gün sayısı bunun üstündeki olaya göre hesaplanır.")
        else:
            st.warning("Model dosyaları veya metadata eksik. Tahmin yapılamıyor.")

# ------------------------------------------------------------
# VISUALS
# ------------------------------------------------------------
st.subheader("📈 Gerçek vs Tahmin Grafikleri")
if pred_df is not None and "date" in pred_df.columns:
    pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")

    if "future_max_magnitude_7d" in pred_df.columns and "pred_future_max_magnitude_7d" in pred_df.columns:
        st.markdown("**Max Magnitude (7 gün)**")
        st.line_chart(
            pred_df.set_index("date")[["future_max_magnitude_7d", "pred_future_max_magnitude_7d"]],
            use_container_width=True,
        )

    if "days_to_next_major_event" in pred_df.columns and "pred_days_to_next_major_event" in pred_df.columns:
        st.markdown("**Days to Next Major Event**")
        st.line_chart(
            pred_df.set_index("date")[["days_to_next_major_event", "pred_days_to_next_major_event"]],
            use_container_width=True,
        )
else:
    st.info("predictions_table_onur.csv bulunamadı.")

# ------------------------------------------------------------
# LIVE MAP
# ------------------------------------------------------------
st.subheader("🗺️ Canlı Deprem Haritası")

if pred_df is not None and {"latitude", "longitude"}.issubset(pred_df.columns):
    map_df = pred_df.dropna(subset=["latitude", "longitude"]).copy()
    map_df["magnitude"] = map_df.get("magnitude", 0).fillna(0)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_radius=5000,
        get_fill_color="[255, 100, 100, 140]",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=map_df["latitude"].mean(),
        longitude=map_df["longitude"].mean(),
        zoom=4,
        pitch=0,
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Mag: {magnitude}"}))
else:
    st.info("Harita için latitude/longitude kolonları bulunamadı.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown("✅ **Deprem Tahmin Uygulaması - Streamlit UI**")
