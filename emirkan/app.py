import streamlit as st
import pandas as pd
from final_decision import run_pipeline

# Sayfa Ayarları
st.set_page_config(page_title="AI Sismik Analiz Paneli", layout="wide")

st.title("🌍 Deprem Risk Analiz ve Erken Uyarı Sistemi")
st.markdown("---")

# Yan Panel (Inputs)
st.sidebar.header("📊 Model Girdileri")

st.sidebar.subheader("Rabia'nın Model Verileri")
risk_score = st.sidebar.slider("Risk Skoru", 0.0, 1.0, 0.85)
risk_prob = st.sidebar.slider("Risk Olasılığı", 0.0, 1.0, 0.76)
region = st.sidebar.selectbox("Bölge Seçiniz", ["Marmara", "Ege", "Akdeniz", "Doğu Anadolu"])

st.sidebar.subheader("Onur'un Model Verileri")
days = st.sidebar.number_input("Tahmini Gün (Days to Event)", 1, 60, 12)
mag = st.sidebar.slider("Beklenen Şiddet (Magnitude)", 2.0, 8.0, 5.8)
conf = st.sidebar.slider("Model Güven Skoru", 0.0, 1.0, 0.78)

# Pipeline'ı Çalıştır
if st.button("Analizi Başlat"):
    # Verileri hazırla
    rabia_in = {"grid_id": "test_id", "region": region, "risk_score": risk_score, "risk_probability": risk_prob}
    onur_in = {"days_to_event": days, "predicted_max_magnitude": mag, "confidence_score": conf}
    
    # Motorları döndür
    results = run_pipeline(rabia_in, onur_in)
    raw = results["raw_data"]

    # --- Görselleştirme ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Alert Level", raw["alert_level"])
    col2.metric("Final Score", round(raw["final_score"], 2))
    col3.metric("Consistency", results["raw_data"]["consistency_check"]["consistency_score"])

    # Rapor Formatları (Sekmeler)
    st.markdown("### 📝 Analiz Raporları")
    tab1, tab2, tab3 = st.tabs(["Teknik Rapor", "Halk Bilgilendirme", "Kurum/AFAD"])
    
    with tab1:
        st.info(results["technical_report"])
    with tab2:
        st.success(results["public_report"])
    with tab3:
        st.warning(results["institutional_report"])

    # Teknik Detaylar (Expandable)
    with st.expander("🔬 Teknik Detayları Gör"):
        st.json(results)
