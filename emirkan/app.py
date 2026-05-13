import os
import sys
import json
import streamlit as st

# --- 1. KLASÖR (PATH) AYARI ---
# emirkan klasörü içindeyken üst klasördeki (onur, rabia) dosyalara ulaşabilmek için
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from emirkan.final_decision import run_pipeline

# --- 2. SAYFA AYARLARI & GÖRSELLİK ---
st.set_page_config(page_title="Alert Intelligence | Sismik Analiz", layout="wide", page_icon="🚨")

col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/1684/1684425.png", width=80) 
with col_title:
    st.title("Alert Intelligence: Real-Time Earthquake App")
    st.markdown("*Yapay Zeka Destekli Deprem Risk Analiz ve Erken Uyarı Sistemi*")
st.markdown("---")


# --- 3. VERİ ÇEKME FONKSİYONLARI ---
def get_rabia_data():
    # Rabia'nın API'si düzelene kadar sistemi besleyecek Mock Veri
    return {
        "grid_id": "34_29",
        "region": "Marmara",
        "risk_score": 0.85,
        "risk_probability": 0.76
    }

def get_onur_data():
    # Onur'un JSON dosyasını okuyan bölüm
    dosya_yolu = os.path.join(parent_dir, "onur", "prediction_output.json")
    if os.path.exists(dosya_yolu):
        with open(dosya_yolu, "r", encoding="utf-8") as f:
            veri = json.load(f)
        return {
            "days_to_event": float(veri.get("days to event", 5.0)),
            "predicted_max_magnitude": float(veri.get("predicted max magnitude", 5.0)),
            "confidence_score": float(veri.get("confidence score", 0.70))
        }
    else:
        st.sidebar.error("❌ Onur'un JSON dosyası bulunamadı! Lütfen önce Onur'un modelini çalıştırın.")
        # Çökmemesi için varsayılan değer dönüyoruz
        return {"days_to_event": 0.0, "predicted_max_magnitude": 0.0, "confidence_score": 0.0}


# --- 4. YAN PANEL (SIDEBAR) & VERİ GİRİŞİ ---
st.sidebar.header("📥 Veri Kaynağı")
source = st.sidebar.radio("Veri Nereden Gelsin?", ["Sistemden Çek (Canlı JSON)", "Manuel Slider (Test)"])

if source == "Sistemden Çek (Canlı JSON)":
    loading = st.sidebar.empty()
    loading.info("⏳ JSON verileri okunuyor...")
    
    rabia_in = get_rabia_data()
    onur_in = get_onur_data()
    
    loading.empty()
    st.sidebar.success("✅ Veriler anında yüklendi!")
    st.sidebar.warning("Not: Rabia'nın verisi şimdilik Mock(Sahte) olarak geliyor.")

else:
    st.sidebar.subheader("Manuel Test Verileri")
    st.sidebar.markdown("**🧠 Rabia'nın Modeli**")
    grid = st.sidebar.text_input("Grid ID", "34_29")
    region = st.sidebar.selectbox("Bölge", ["Marmara", "Ege", "Akdeniz", "Doğu Anadolu", "KAF"])
    risk_score = st.sidebar.slider("Risk Skoru (Rabia)", 0.0, 1.0, 0.85)
    risk_prob = st.sidebar.slider("Risk Olasılığı", 0.0, 1.0, 0.76)
    
    st.sidebar.markdown("**⏱️ Onur'un Modeli**")
    days = st.sidebar.slider("Tahmini Gün (Onur)", 1.0, 60.0, 5.0)
    mag = st.sidebar.slider("Beklenen Şiddet", 1.0, 9.0, 5.8)
    conf = st.sidebar.slider("Model Güven Skoru", 0.0, 1.0, 0.78)
    
    rabia_in = {"grid_id": grid, "region": region, "risk_score": float(risk_score), "risk_probability": float(risk_prob)}
    onur_in = {"days_to_event": float(days), "predicted_max_magnitude": float(mag), "confidence_score": float(conf)}


# --- 5. TUTARLILIK MOTORUNU ÇALIŞTIR ---
try:
    results = run_pipeline(rabia_in, onur_in)
    raw = results.get("raw_data", {})
    consistency_data = raw.get("consistency_check", {})
    calculated_final_score = raw.get('final_score', 0)
except Exception as e:
    st.error(f"🚨 Pipeline Motoru Çalışırken Hata Oluştu: {e}")
    raw, consistency_data, calculated_final_score, results = {}, {}, 0, {}


# --- 6. ANA EKRAN GÖRSELLEŞTİRMELERİ ---
st.markdown("### 📊 Anlık Sismik Risk Metrikleri")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📌 Bölge", f"{rabia_in.get('region')} ({rabia_in.get('grid_id')})")
col2.metric("⚠️ Uyarı Seviyesi", raw.get('alert_level', 'BELİRSİZ'))
col3.metric("🔥 Final Risk Skoru", f"{calculated_final_score:.2f}")
col4.metric("⚖️ Tutarlılık Skoru", f"{consistency_data.get('consistency_score', 0):.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# Tutarlılık Durumu
is_consistent = consistency_data.get("is_consistent", True)
status_text = consistency_data.get("status", "Bilinmiyor")

if is_consistent:
    st.success(f"✅ **Modeller Birbiriyle Tutarlı:** {status_text}")
else:
    st.error(f"❌ **Modeller Arasında Çelişki Var!** Durum: {status_text}")

st.markdown("<br>", unsafe_allow_html=True)

# Sisteme Giren Ham Veriler
col_r, col_o = st.columns(2)
with col_r:
    st.info("🧠 **Sınıflandırma Modeli (Rabia)**")
    st.write(f"- **Ham Risk Skoru:** {rabia_in.get('risk_score')}")
    st.write(f"- **Risk Olasılığı:** {rabia_in.get('risk_probability')}")
    
with col_o:
    st.warning("⏱️ **Tahmin Modeli (Onur)**")
    st.write(f"- **Beklenen Maksimum Şiddet:** {onur_in.get('predicted_max_magnitude')}")
    st.write(f"- **Kalan Gün Tahmini:** {onur_in.get('days_to_event')} gün")
    st.write(f"- **Model Güven Skoru:** {onur_in.get('confidence_score')}")

st.markdown("---")
st.error(f"🚨 **SİSTEM ÖNERİSİ:** {raw.get('recommended_action', 'Veri bekleniyor...')}")
st.markdown("---")

# Raporlar Sekmesi
st.markdown("### 📝 Analiz Raporları")
tab1, tab2, tab3 = st.tabs(["🔬 Teknik Rapor", "📢 Kamuoyu Duyurusu", "🏢 AFAD Dili"])

with tab1:
    st.info(results.get("technical_report", "Rapor üretilemedi."))
with tab2:
    st.success(results.get("public_report", "Rapor üretilemedi."))
with tab3:
    st.warning(results.get("institutional_report", "Rapor üretilemedi."))

    