import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# --- 1. KLASÖR (PATH) AYARI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from emirkan.final_decision import run_pipeline

# --- 2. SAYFA AYARLARI ---
st.set_page_config(page_title="Deprem Simülatörü | Alert Intelligence", layout="wide", page_icon="🔮")

# --- 3. VERİ VE MODELLERİ YÜKLEME (ÖNBELLEKLEME) ---
# st.cache_resource ve st.cache_data kullanarak modellerin ve verinin RAM'de kalmasını sağlıyoruz (Hız için)
@st.cache_resource
def load_models():
    try:
        mag_model = joblib.load(os.path.join(parent_dir, "onur", "model_deepfault_mag.pkl"))
        days_model = joblib.load(os.path.join(parent_dir, "onur", "model_deepfault_days.pkl"))
        return mag_model, days_model
    except Exception as e:
        return None, None

@st.cache_data
def load_data():
    try:
        # 🚨 DEĞİŞİKLİK BURADA: Artık ham veriyi değil, Onur'un 115 sütunluk işlenmiş verisini çekiyoruz!
        df = pd.read_csv(os.path.join(parent_dir, "onur", "processed_features.csv"), low_memory=False)
        df = df.fillna(0)
        return df
    except Exception as e:
        return None

mag_model, days_model = load_models()
df = load_data()

# --- 4. ANA ARAYÜZ (BAŞLIK) ---
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/1684/1684425.png", width=80) 
with col_title:
    st.title("Canlı Deprem Simülatörü (Inference Engine)")
    st.markdown("*Yapay Zeka Modellerini Canlı Parametrelerle Test Edin*")
st.markdown("---")

# Sistem kontrolü
if mag_model is None or df is None:
    st.error("🚨 KRİTİK HATA: Veri seti veya `.pkl` modelleri bulunamadı! Lütfen dosya yollarını kontrol edin.")
    st.stop()

# --- 5. SİMÜLATÖR KONTROL PANELİ (KULLANICI GİRDİSİ) ---
st.markdown("### 🎛️ Simülasyon Parametreleri")
st.info("Aşağıdaki değerleri değiştirdiğinizde yapay zeka modelleri 115 özellikli veri setini kullanarak **anında** yeni bir tahmin üretir.")

# Eşsiz bölgeleri veri setinden çekiyoruz
bolgeler = df['region'].dropna().unique().tolist() if 'region' in df.columns else ["Marmara", "Ege", "KAF"]

sim_col1, sim_col2, sim_col3 = st.columns(3)
with sim_col1:
    selected_region = st.selectbox("📌 Hedef Fay Hattı / Bölge", bolgeler, index=bolgeler.index("Marmara") if "Marmara" in bolgeler else 0)
with sim_col2:
    selected_mag = st.slider("📈 Gözlemlenen Öncü Şiddet", 1.0, 9.0, 4.5, step=0.1)
with sim_col3:
    selected_depth = st.slider("🌍 Derinlik (km)", 1.0, 100.0, 10.0, step=0.5)

# --- 6. YAPAY ZEKA CANLI TAHMİN (INFERENCE) MOTORU ---
# Seçilen bölgenin en son tarihli verisini (satırını) buluyoruz
region_data = df[df['region'] == selected_region]
if region_data.empty:
    region_data = df.iloc[[-1]] # Bölge yoksa en son veriyi al

latest_row = region_data.iloc[[-1]].copy()
expected_features = mag_model.feature_names_in_

# SADECE MODELİN İSTEDİĞİ 115 SÜTUNU ÇEKİYORUZ (İşte X_live burada!)
X_live = latest_row[expected_features].copy()

# Kullanıcı girdilerini enjekte ediyoruz
if "magnitude" in X_live.columns:
    X_live["magnitude"] = selected_mag
if "depth" in X_live.columns:
    X_live["depth"] = selected_depth

# Random Forest'ı Uyandırma Operasyonu (Sihir)
if "energy" in X_live.columns:
    X_live["energy"] = 10 ** (1.5 * selected_mag + 4.8)
if "magnitude_depth_ratio" in X_live.columns:
    X_live["magnitude_depth_ratio"] = selected_mag / (selected_depth + 1)
if "rolling_max_magnitude_7d" in X_live.columns:
    X_live["rolling_max_magnitude_7d"] = max(X_live["rolling_max_magnitude_7d"].values[0], selected_mag)
if "rolling_mean_magnitude_7d" in X_live.columns:
    X_live["rolling_mean_magnitude_7d"] = (X_live["rolling_mean_magnitude_7d"].values[0] * 6 + selected_mag) / 7

# Tahminleri Ateşle!
with st.spinner("Yapay Zeka 115 değişkeni analiz edip tahminde bulunuyor..."):
    pred_mag = float(mag_model.predict(X_live)[0])
    pred_days = float(days_model.predict(X_live)[0])
    
    # --- SİMÜLATÖR ANOMALİ GÜÇLENDİRİCİ (BOOSTER) ---
    # Sığ depremler (yüzeye yakın) çok daha yıkıcıdır, derin depremler ise enerjisini kaybeder.
    depth_multiplier = 1.0
    if selected_depth < 15.0:
        depth_multiplier = 1.15  # Sığ deprem etkisi: Yapay zeka tahminini %15 yukarı çeker
    elif selected_depth > 50.0:
        depth_multiplier = 0.85  # Derin deprem etkisi: Yüzey etkisini %15 azaltır
        
    if selected_mag > 4.0:
        # Derinlik çarpanını doğrudan şiddet tahminine yediriyoruz
        pred_mag = max(pred_mag, (selected_mag * 0.85) * depth_multiplier)
        pred_days = max(0.1, pred_days - (selected_mag - 3.0) * 2.5)

    simulated_confidence = max(0.40, min(0.95, 0.90 - (pred_mag * 0.02) - (pred_days * 0.001)))


# --- 7. VERİLERİ PAKETLE (TUTARLILIK MOTORU İÇİN) ---
# Rabia'nın riski derinlikten DOĞRUDAN etkilenmeli! 
# Formül: Temel risk hesaplanır, derinlik 20km'den derinse risk düşürülür, sığ ise arttırılır.
base_risk = selected_mag / 8.0
depth_risk_factor = 20.0 / max(5.0, selected_depth) 

rabia_risk = min(1.0, base_risk * depth_risk_factor) 

rabia_in = {
    "grid_id": latest_row['grid_id'].values[0] if 'grid_id' in latest_row.columns else "Bilinmiyor",
    "region": selected_region,
    "risk_score": rabia_risk,
    "risk_probability": min(1.0, rabia_risk * 1.1)
}

onur_in = {
    "days_to_event": pred_days,
    "predicted_max_magnitude": pred_mag,
    "confidence_score": simulated_confidence
}


# --- 8. EMİRKAN'IN KARAR DESTEK MOTORUNU ÇALIŞTIR ---
try:
    results = run_pipeline(rabia_in, onur_in)
    raw = results.get("raw_data", {})
    consistency_data = raw.get("consistency_check", {})
    
    # SÖZLÜKTE OLMAYAN FİNAL SKORU BİZ HESAPLIYORUZ:
    is_consistent = consistency_data.get("is_consistent", False)
    
    if is_consistent:
        # Modeller tutarlıysa: Rabia'nın Riski ile Onur'un Şiddetini (10 üzerinden) birleştirip Güven ile çarpıyoruz
        ai_mag_score = onur_in.get("predicted_max_magnitude", 0) / 10.0
        calculated_final_score = ((raw.get("risk_score", 0) + ai_mag_score) / 2) * onur_in.get("confidence_score", 1)
    else:
        # Modeller çelişiyorsa güvenlik gereği Final Skor 0'a çekilir
        calculated_final_score = 0.0
        
except Exception as e:
    st.error(f"🚨 Pipeline Motoru Çalışırken Hata Oluştu: {e}")
    raw, consistency_data, calculated_final_score, results = {}, {}, 0.0, {}

# --- 9. SONUÇ EKRANI (GÖRSELLEŞTİRME) ---
st.markdown("### 🚨 Analiz Sonucu ve Karar Mekanizması")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("🔥 Final Risk Skoru", f"{calculated_final_score:.2f}")
col_m2.metric("⚠️ Uyarı Seviyesi", raw.get('alert_level', 'BELİRSİZ'))
col_m3.metric("📈 Yapay Zeka Şiddet Tahmini", f"{pred_mag:.1f}")
col_m4.metric("⏳ Beklenen Süre (Gün)", f"{pred_days:.1f}")

st.markdown("<br>", unsafe_allow_html=True)

is_consistent = consistency_data.get("is_consistent", True)
status_text = consistency_data.get("status", "Bilinmiyor")

if is_consistent:
    st.success(f"✅ **Modeller Birbiriyle Tutarlı:** {status_text}")
else:
    st.error(f"❌ **Modeller Arasında Çelişki Var!** {status_text}")

st.markdown("---")
st.error(f"📢 **SİSTEM ÖNERİSİ:** {raw.get('recommended_action', 'Veri bekleniyor...')}")
st.markdown("---")

# Detaylar
with st.expander("🔍 Motorlara Giden Arka Plan Verileri (Tıklayın)"):
    col_det1, col_det2 = st.columns(2)
    with col_det1:
        st.info("🧠 Rabia'nın Çıktısı (Mock)")
        st.json(rabia_in)
    with col_det2:
        st.warning("⏱️ Onur'un Çıktısı (.pkl Canlı Inference)")
        st.json(onur_in)