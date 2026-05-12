import streamlit as st
import pandas as pd
import json
import pydeck as pdk

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="Earthquake Early Warning App",
    layout="wide",
    page_icon="🌍"
)

st.title("🌍 EARLY WARNING EARTHQUAKE APP")
st.markdown("Bu uygulama **deprem tahmin sonuçlarını**, **model meta verilerini** ve **harita tabanlı analizi** sunar.")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_predictions():
    return pd.read_csv("onur/predictions_table_onur.csv", low_memory=False)

@st.cache_data
def load_metadata():
    with open("onur/OUTPUTS_onur.json", "r", encoding="utf-8") as f:
        return json.load(f)

try:
    pred_df = load_predictions()
    metadata = load_metadata()
except Exception:
    pred_df = None
    metadata = None

# ------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------
st.sidebar.header("🔎 Filtreleme")

if pred_df is not None:
    if "date" in pred_df.columns:
        pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce")

    if "region" in pred_df.columns:
        regions = ["Tümü"] + sorted(pred_df["region"].dropna().unique().tolist())
        selected_region = st.sidebar.selectbox("Bölge Seç", regions)
        if selected_region != "Tümü":
            pred_df = pred_df[pred_df["region"] == selected_region]

    if "date" in pred_df.columns:
        min_date, max_date = pred_df["date"].min(), pred_df["date"].max()
        date_range = st.sidebar.date_input("Tarih Aralığı", [min_date, max_date])
        if len(date_range) == 2:
            pred_df = pred_df[
                (pred_df["date"] >= pd.to_datetime(date_range[0])) &
                (pred_df["date"] <= pd.to_datetime(date_range[1]))
            ]

# ------------------------------------------------------------
# METADATA + MODEL PARAMS
# ------------------------------------------------------------
st.subheader("📌 Model Meta Verileri")

if metadata:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Versiyon", metadata.get("model_version", "-"))
    col2.metric("Veri Satır Sayısı", metadata.get("rows", "-"))
    col3.metric("Feature Sayısı", metadata.get("feature_count", "-"))
    col4.metric("Üretim Tarihi", metadata.get("generated_at", "-"))

    st.json(metadata)

    st.subheader("⚙️ Model Parametrelerini Seç ve İncele")
    params = metadata.get("best_params", {})
    if params:
        target_options = list(params.keys())
        selected_target = st.selectbox("Parametreleri Görüntüle (Target Seç)", target_options)
        st.json(params.get(selected_target, {}))
    else:
        st.warning("best_params bilgisi bulunamadı.")
else:
    st.warning("OUTPUTS_onur.json bulunamadı. Önce main_onur.py çalıştırmalısın.")

# ------------------------------------------------------------
# MODEL VERSION COMPARISON
# ------------------------------------------------------------
st.subheader("🔄 Model Versiyon Kıyaslama")

if metadata and "model_versions" in metadata:
    version_df = pd.DataFrame(metadata["model_versions"])
    st.dataframe(version_df)
else:
    st.info("Model versiyon kıyaslama için OUTPUTS_onur.json içine `model_versions` listesi ekleyebilirsin.")

# ------------------------------------------------------------
# DATA TABLE
# ------------------------------------------------------------
st.subheader("📋 Tahmin + Gerçek Veri Tablosu")

if pred_df is None:
    st.error("predictions_table_onur.csv bulunamadı. Önce main_onur.py çalıştırmalısın.")
else:
    st.dataframe(pred_df.tail(50))

    csv_data = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Tahmin CSV İndir",
        data=csv_data,
        file_name="predictions_table_onur.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------
# PREDICTION JSON OUTPUT (LATEST ROW)
# ------------------------------------------------------------
st.subheader("🧾 Tahmin JSON Çıktısı (Son Kayıt)")

if pred_df is not None and len(pred_df) > 0:
    latest = pred_df.tail(1).to_dict(orient="records")[0]
    st.json(latest)
else:
    st.warning("Tahmin verisi bulunamadı.")

# ------------------------------------------------------------
# VISUALS
# ------------------------------------------------------------
st.subheader("📊 Gerçek vs Tahmin Grafikler")

if pred_df is not None and "date" in pred_df.columns:
    if "future_max_magnitude_7d" in pred_df.columns and "pred_future_max_magnitude_7d" in pred_df.columns:
        st.markdown("**Max Magnitude (7 gün)**")
        st.line_chart(
            pred_df.set_index("date")[["future_max_magnitude_7d", "pred_future_max_magnitude_7d"]],
            use_container_width=True
        )

    if "days_to_next_major_event" in pred_df.columns and "pred_days_to_next_major_event" in pred_df.columns:
        st.markdown("**Days to Next Major Event**")
        st.line_chart(
            pred_df.set_index("date")[["days_to_next_major_event", "pred_days_to_next_major_event"]],
            use_container_width=True
        )

# ------------------------------------------------------------
# CONFIDENCE SCORE GRAPH
# ------------------------------------------------------------
st.subheader("📈 Confidence Score Grafiği")

confidence_cols = [c for c in pred_df.columns if "confidence" in c.lower()] if pred_df is not None else []
if pred_df is not None and confidence_cols:
    st.line_chart(pred_df.set_index("date")[confidence_cols], use_container_width=True)
else:
    st.info("Confidence skoru için CSV içinde `confidence` içeren kolon bulunamadı.")

# ------------------------------------------------------------
# LIVE MAP (PYDECK)
# ------------------------------------------------------------
st.subheader("🗺️ Canlı Deprem Haritası")

if pred_df is not None and {"latitude", "longitude"}.issubset(pred_df.columns):
    map_df = pred_df.dropna(subset=["latitude", "longitude"]).copy()
    map_df["magnitude"] = map_df["magnitude"].fillna(0)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_radius=5000,
        get_fill_color="[255, 100, 100, 140]",
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=map_df["latitude"].mean(),
        longitude=map_df["longitude"].mean(),
        zoom=4,
        pitch=0
    )

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Mag: {magnitude}"}))
else:
    st.info("Harita için latitude/longitude kolonları bulunamadı.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.markdown("✅ **Deprem Tahmin Uygulaması - Streamlit UI**")