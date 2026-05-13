import os
import sys
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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

# --- 5.5 DİNAMİK FAY HARİTASI VE ETKİ ALANI ŞOVU ---
st.markdown("### 🗺️ Bölgesel Etki ve Risk Haritası")
# Bölgelerin ortalama koordinatları (Jürilik şov kısmı)
koordinatlar = {
    "KAF": {"lat": 40.8, "lon": 33.0, "name": "Kuzey Anadolu Fayı"},
    "DAF": {"lat": 38.2, "lon": 38.8, "name": "Doğu Anadolu Fayı"},
    "BAF": {"lat": 38.5, "lon": 28.0, "name": "Batı Anadolu Fayı"},
    "Marmara": {"lat": 40.7, "lon": 28.5, "name": "Marmara Bölgesi"},
    "Ege": {"lat": 38.0, "lon": 26.0, "name": "Ege Bölgesi"}
}

# Seçilen bölgenin koordinatını al (Yoksayılan olarak Ankara civarı)
curr_lat = koordinatlar.get(selected_region, {"lat": 39.0})["lat"]
curr_lon = koordinatlar.get(selected_region, {"lon": 35.0})["lon"]
bolge_adi = koordinatlar.get(selected_region, {"name": selected_region})["name"]

# Haritayı oluştur (Şiddet arttıkça daire büyür ve kızarır)
# Dairenin büyüklüğü şiddetin üstel (exponential) haliyle artar ki görsel etki vursun
map_fig = px.scatter_mapbox(
    lat=[curr_lat],
    lon=[curr_lon],
    size=[selected_mag ** 3],  
    color=[selected_mag],
    color_continuous_scale="Reds",
    range_color=[1.0, 9.0],
    hover_name=[f"Simülasyon Merkezi: {bolge_adi}"],
    zoom=5.5,
    mapbox_style="carto-darkmatter",
    height=400
)
map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_showscale=False)
st.plotly_chart(map_fig, use_container_width=True)
st.info(f"📍 **{bolge_adi}** üzerinde {selected_mag} büyüklüğünde ve {selected_depth} km derinliğinde bir senaryo simüle ediliyor.")


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

# --- 10. SUNUM İÇİN BİLİMSEL GRAFİKLER ---
st.markdown("---")
st.markdown("### 📊 Sismik Analiz ve Proje Mantığı")

grafik_col1, grafik_col2 = st.columns(2)

with grafik_col1:
    st.markdown("**1. Tarihsel Deprem Dağılımı (Seçilen Bölge)**")
    st.markdown("Yapay zekanın bu bölgedeki geçmiş depremlerin derinlik ve şiddet korelasyonunu nasıl okuduğunu gösterir.")
    # Sadece o bölgenin geçmiş verilerini çiziyoruz
    if not region_data.empty and "magnitude" in region_data.columns and "depth" in region_data.columns:
        fig_scatter = px.scatter(
            region_data, x="magnitude", y="depth", 
            color="magnitude", color_continuous_scale="Viridis",
            labels={"magnitude": "Şiddet (Mw)", "depth": "Derinlik (km)"},
            title=f"{selected_region} Bölgesi Geçmiş Veri Kümesi"
        )
        fig_scatter.update_layout(yaxis_autorange="reversed") # Derinlik aşağı doğru artar
        # Simüle edilen noktayı Kırmızı X olarak ekliyoruz (Çok havalı sunum detayı)
        fig_scatter.add_scatter(x=[selected_mag], y=[selected_depth], mode='markers', 
                                marker=dict(size=15, color='red', symbol='x'), name='CANLI SİMÜLASYON')
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Bu bölge için yeterli tarihsel veri bulunamadı.")

with grafik_col2:
    st.markdown("**2. Tutarlılık Motoru (Radar Analizi)**")
    st.markdown("Karar destek sistemimizin riski belirlerken hangi parametrelere ne kadar ağırlık verdiğinin canlı haritası.")
    
    # Radar Grafiği için verileri 0-1 arasına normalize ediyoruz (Görsel şov)
    categories = ['Sismik Enerji\n(Magnitude)', 'Yüzey Etkisi\n(Sığlık)', 'AI Güven Skoru', 'Rabia Risk Skoru', 'Final Karar Skoru']
    
    # Değerler
    mag_norm = selected_mag / 10.0
    depth_norm = max(0, 1 - (selected_depth / 100.0)) # Sığ ise 1'e yakın olur
    
    values = [mag_norm, depth_norm, simulated_confidence, rabia_risk, calculated_final_score]
    
    fig_radar = go.Figure(data=go.Scatterpolar(
      r=values,
      theta=categories,
      fill='toself',
      line_color='darkorange'
    ))
    fig_radar.update_layout(
      polar=dict(
        radialaxis=dict(visible=True, range=[0, 1])
      ),
      showlegend=False,
      title="Anlık Karar Karakteristiği"
    )
    st.plotly_chart(fig_radar, use_container_width=True)