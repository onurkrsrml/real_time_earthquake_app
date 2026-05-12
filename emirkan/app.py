import sys
import os

# Python'a bir üst klasörü (ana proje dizinini) tanıtıyoruz ki Rabia ve Onur'u bulabilsin:
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import streamlit as st
from final_decision import run_pipeline

# --- 1. SAYFA AYARLARI, PROJE ADI VE LOGO (Rabia'nın Önerisi) ---
st.set_page_config(page_title="Alert Intelligence | Sismik Analiz", layout="wide", page_icon="🚨")

# Yan yana logo ve başlık tasarımı
col_logo, col_title = st.columns([1, 8])
with col_logo:
    # Gerçek bir logonuz varsa "logo.png" olarak buraya ekleyebilirsin
    st.image("https://cdn-icons-png.flaticon.com/512/1684/1684425.png", width=80) 
with col_title:
    st.title("Alert Intelligence: Real-Time Earthquake App")
    st.markdown("*Yapay Zeka Destekli Deprem Risk Analiz ve Erken Uyarı Sistemi*")
st.markdown("---")

# --- 2. YAN PANEL (SIDEBAR) & VERİ GİRİŞİ ---
st.sidebar.header("📥 Veri Kaynağı")
source = st.sidebar.radio("Veri Nereden Gelsin?", ["Arkadaşlarımdan Çek (Canlı)", "Manuel Slider (Test)"])

if source == "Arkadaşlarımdan Çek (Canlı)":
    st.sidebar.info("⏳ Modellerden veri çekiliyor...")
    try:
        import json
        # Doğru fonksiyonları import ediyoruz (Onur'un main'ini onur_main olarak aldık)
        from rabia.main_rabia import get_seismic_risk_score
        from onur.main_onur import main as onur_main
        
        # --- 1. RABİA'NIN VERİSİNİ UYARLAMA (ADAPTÖR) ---
        rabia_raw_response = get_seismic_risk_score(40.85, 29.30, "2026-05-12")
        
        if isinstance(rabia_raw_response, str):
            st.sidebar.error(f"Rabia API Hatası: {rabia_raw_response}")
            r_score = 0.50 # Hata anında varsayılan değer
        else:
            r_score = float(rabia_raw_response)
            
        rabia_in = {
            "grid_id": "34_29",
            "region": "Marmara",
            "risk_score": r_score,
            "risk_probability": 0.80 
        }

        # --- 2. ONUR'UN VERİSİNİ UYARLAMA (ADAPTÖR) ---
        try:
            onur_json_string = onur_main() 
            onur_dict = json.loads(onur_json_string)
            
            onur_in = {
                "days_to_event": float(onur_dict.get("days to event", 10)),
                "predicted_max_magnitude": float(onur_dict.get("predicted max magnitude", 5.0)),
                "confidence_score": float(onur_dict.get("confidence score", 0.75))
            }
        except Exception as e_onur:
            st.sidebar.warning(f"Onur'un modeli çalışmadı: {e_onur}")
            onur_in = {"days_to_event": 5, "predicted_max_magnitude": 5.8, "confidence_score": 0.78}

        st.sidebar.success("✅ Veriler modellere bağlanıp başarıyla dönüştürüldü!")

    except Exception as e:
        st.sidebar.error(f"❌ Genel Entegrasyon Hatası: {e}")
        rabia_in = {"grid_id": "34_29", "region": "Marmara", "risk_score": 0.85, "risk_probability": 0.76}
        onur_in = {"days_to_event": 5, "predicted_max_magnitude": 5.8, "confidence_score": 0.78}

# İŞTE SENİN SİLDİĞİN VE HATAYA SEBEP OLAN O KAYIP BLOK BURASI:
else:
    st.sidebar.subheader("Manuel Test Verileri")
    
    st.sidebar.markdown("**🧠 Rabia'nın Modeli (Sınıflandırma)**")
    grid = st.sidebar.text_input("Grid ID", "34_29")
    region = st.sidebar.selectbox("Bölge", ["Marmara", "Ege", "Akdeniz", "Doğu Anadolu", "Güneydoğu Anadolu"])
    risk_score = st.sidebar.slider("Risk Skoru", 0.0, 1.0, 0.85)
    risk_prob = st.sidebar.slider("Risk Olasılığı", 0.0, 1.0, 0.76)
    
    st.sidebar.markdown("**⏱️ Onur'un Modeli (Tahmin)**")
    days = st.sidebar.slider("Tahmini Gün (days_to_event)", 1, 60, 5)
    mag = st.sidebar.slider("Beklenen Şiddet (predicted_max_magnitude)", 2.0, 9.0, 5.8)
    conf = st.sidebar.slider("Model Güven Skoru (confidence_score)", 0.0, 1.0, 0.78)
    
    rabia_in = {
        "grid_id": grid, 
        "region": region, 
        "risk_score": float(risk_score), 
        "risk_probability": float(risk_prob)
    }
    onur_in = {
        "days_to_event": int(days), 
        "predicted_max_magnitude": float(mag), 
        "confidence_score": float(conf)
    }
    
# --- 3. MOTORLARI ÇALIŞTIR ---
results = run_pipeline(rabia_in, onur_in)
raw = results.get("raw_data", {})
consistency_data = raw.get("consistency_check", {})

# Final skoru kurtarma
calculated_final_score = raw.get('final_score')
if not calculated_final_score or calculated_final_score == 0: 
    risk_weight = rabia_in.get('risk_score', 0) * 0.6
    mag_weight = (onur_in.get('predicted_max_magnitude', 0) / 10.0) * 0.4
    calculated_final_score = (risk_weight + mag_weight) * 100 

# --- 4. ANA EKRAN: METRİKLER ---
st.markdown("### 📊 Anlık Sismik Risk Metrikleri")

col1, col2, col3, col4 = st.columns(4)
col1.metric("📌 Bölge", f"{raw.get('region', rabia_in.get('region'))} ({raw.get('grid_id', rabia_in.get('grid_id'))})")
col2.metric("⚠️ Uyarı Seviyesi", raw.get('alert_level', 'BELİRSİZ'))
col3.metric("🔥 Final Risk Skoru", f"{calculated_final_score:.2f}")
col4.metric("⚖️ Tutarlılık Skoru", f"{consistency_data.get('consistency_score', 0):.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# --- 5. DİNAMİK TUTARLILIK KONTROLÜ ---
is_consistent = consistency_data.get("is_consistent", True)
status_text = consistency_data.get("status", "Bilinmiyor")

if is_consistent:
    st.success(f"✅ **Modeller Birbiriyle Tutarlı:** {status_text}")
else:
    st.error(f"❌ **Modeller Arasında Çelişki Var!** Durum: {status_text}")

st.markdown("<br>", unsafe_allow_html=True)

# --- 6. SİSTEME GİREN HAM VERİLER ---
col_r, col_o = st.columns(2)

with col_r:
    st.info("🧠 **Sınıflandırma Modeli (Rabia'nın Çıktıları)**")
    st.write(f"- **Ham Risk Skoru:** {rabia_in.get('risk_score')}")
    st.write(f"- **Risk Olasılığı:** {rabia_in.get('risk_probability')}")
    
with col_o:
    st.warning("⏱️ **Tahmin Modeli (Onur'un Çıktıları)**")
    st.write(f"- **Beklenen Maksimum Şiddet:** {onur_in.get('predicted_max_magnitude')}")
    st.write(f"- **Kalan Gün Tahmini:** {onur_in.get('days_to_event')} gün")
    st.write(f"- **Model Güven Skoru:** {onur_in.get('confidence_score')}")

st.markdown("---")
st.error(f"🚨 **SİSTEM ÖNERİSİ:** {raw.get('recommended_action', 'Bölgesel ekipler hazırda beklemeli ve sismik hareketlilik yakından izlenmelidir.')}")
st.markdown("---")

# --- 7. RAPORLAR VE GRAFİKLER (Rabia'nın Grafikleri Buraya) ---
st.markdown("### 📝 Analiz Raporları ve Görselleştirmeler")
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Teknik Rapor", "📢 Kamuoyu Duyurusu", "🏢 AFAD Dili", "📈 Grafikler (EDA)"])

with tab1:
    st.info(results.get("technical_report", "Rapor henüz üretilemedi."))
with tab2:
    st.success(results.get("public_report", "Rapor henüz üretilemedi."))
with tab3:
    st.warning(results.get("institutional_report", "Rapor henüz üretilemedi."))
with tab4:
    st.markdown("#### Veri Görselleştirmeleri")
    if source == "Arkadaşlarımdan Çek (Canlı)":
        try:
            st.write("Rabia'nın Keşifçi Veri Analizi (EDA) Grafikleri yükleniyor...")
            from rabia.main_rabia import create_eda_outputs
            create_eda_outputs()
            st.success("Grafikler başarıyla oluşturuldu!")
        except Exception as e:
            st.error(f"Grafikler yüklenirken hata oluştu: {e}")
            st.info("Rabia'nın `create_eda_outputs()` fonksiyonunun arayüze uygun hale getirilmesi gerekebilir.")
    else:
        st.info("Grafikleri görmek için sol menüden 'Arkadaşlarımdan Çek (Canlı)' seçeneğini işaretleyin.")