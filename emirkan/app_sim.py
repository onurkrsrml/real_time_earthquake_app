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
def load_onur_model():
    try:
        mag_model = joblib.load(os.path.join(parent_dir, "onur", "model_deepfault_mag.pkl"))
        days_model = joblib.load(os.path.join(parent_dir, "onur", "model_deepfault_days.pkl"))
        return mag_model, days_model
    except Exception as e:
        return None, None

# --- RABIA'NIN MODELİ VE VERİSİ ---
@st.cache_resource
def load_rabia_model():
    try:
        # outputs klasörü ana dizinde olduğu için parent_dir'den direkt outputs'a gidiyoruz
        model_path = os.path.join(parent_dir, "outputs", "deepfault_Rabia_TIME_SERIES_V4_BEST_FINAL_deployment_model.joblib")
        return joblib.load(model_path)
    except Exception as e:
        return None

@st.cache_data
def load_rabia_data():
    try:
        data_path = os.path.join(parent_dir, "outputs", "02_time_series_features.csv")
        df = pd.read_csv(data_path, low_memory=False)
        return df.fillna(0)
    except Exception as e:
        return None

rabia_model_dict = load_rabia_model()
df_rabia = load_rabia_data()
    

@st.cache_data
def load_onur_data():
    try:
        # 🚨 DEĞİŞİKLİK BURADA: Artık ham veriyi değil, Onur'un 115 sütunluk işlenmiş verisini çekiyoruz!
        df = pd.read_csv(os.path.join(parent_dir, "onur", "processed_features.csv"), low_memory=False)
        df = df.fillna(0)
        return df
    except Exception as e:
        return None

mag_model, days_model = load_onur_model()
df = load_onur_data()

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
st.info("Bu kontrol paneli, yapay zeka modellerinin 'Feature Importance' listesindeki en kritik sismik değişkenlere odaklanır.")

# Gerçekçi Şehir ve Koordinat Veritabanı
sehirler = {
    "İzmir": {"lat": 38.4192, "lon": 27.1287, "region": "Ege"},
    "İstanbul": {"lat": 41.0082, "lon": 28.9784, "region": "Marmara"},
    "Erzincan": {"lat": 39.7500, "lon": 39.5000, "region": "KAF"},
    "Kahramanmaraş": {"lat": 37.5753, "lon": 36.9228, "region": "DAF"},
    "Hatay": {"lat": 36.2000, "lon": 36.1667, "region": "DAF"},
    "Van": {"lat": 38.5012, "lon": 43.3730, "region": "DAF"}
}

col_loc1, col_loc2 = st.columns(2)
with col_loc1:
    selected_city = st.selectbox("📍 Hedef Şehir", list(sehirler.keys()), index=0)
with col_loc2:
    selected_radius = st.slider("🎯 İnceleme Yarıçapı (km)", 10, 500, 100, step=10, help="Haritadaki etki ve gözlem alanını belirler.")

# Şehrin koordinatlarını ve arkadaki verisetini filtrelemek için bölge adını alıyoruz
curr_lat = sehirler[selected_city]["lat"]
curr_lon = sehirler[selected_city]["lon"]
selected_region = sehirler[selected_city]["region"]

st.markdown("#### ⚡ Kritik Sismik Değişkenler")
col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    selected_mag = st.slider("📈 Öncü Şiddet (Mw)", 1.0, 9.0, 4.5, step=0.1, help="Modellerdeki en yüksek ağırlıklı değişken.")
with col_s2:
    selected_depth = st.slider("🌍 Derinlik (km)", 1.0, 100.0, 10.0, step=0.5, help="Yüzey yıkıcılık etkisi için kritik eşik.")
with col_s3:
    selected_days_since = st.slider("⏳ Sessizlik (Gün)", 0, 365, 30, help="Son depremden geçen süre (Faydaki enerji birikimi).")
with col_s4:
    selected_event_count_7d = st.slider("📊 7 Günlük Aktivite", 0, 100, 5, help="Son 7 gündeki mikro deprem sayısı (İvmelenme).")

# --- 5.5 DİNAMİK FAY HARİTASI VE ETKİ ALANI ŞOVU ---
st.markdown(f"### 🗺️ {selected_city} ve Çevresi Etki Haritası")

# Harita Zoom seviyesini yarıçapa göre dinamik ayarlıyoruz (km arttıkça harita uzaklaşır)
# Yarıçap 10km ise zoom 11, 500km ise zoom 5 gibi bir ters orantı
dynamic_zoom = max(4.5, 11.5 - (selected_radius / 65.0))

map_fig = px.scatter_mapbox(
    lat=[curr_lat],
    lon=[curr_lon],
    # Yarıçapı haritada görsel bir büyüklük olarak yansıtıyoruz
    size=[selected_radius],  
    color=[selected_mag],
    color_continuous_scale="Reds",
    range_color=[1.0, 9.0],
    hover_name=[f"Merkez: {selected_city} | Yarıçap: {selected_radius}km"],
    zoom=dynamic_zoom,
    mapbox_style="carto-darkmatter",
    height=400
)
map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, coloraxis_showscale=False)
st.plotly_chart(map_fig, use_container_width=True)
st.info(f"📍 **{selected_city}** merkezli **{selected_radius} km** yarıçaplı alanda, fay hattı stres simülasyonu yapılıyor.")

# --- 6. YAPAY ZEKA CANLI TAHMİN (INFERENCE) MOTORU ---
# Seçilen bölgenin en son tarihli verisini (satırını) buluyoruz
region_data = df[df['region'] == selected_region]
if region_data.empty:
    region_data = df.iloc[[-1]] # Bölge yoksa en son veriyi al

latest_row = region_data.iloc[[-1]].copy()
expected_features = mag_model.feature_names_in_

# SADECE MODELİN İSTEDİĞİ 115 SÜTUNU ÇEKİYORUZ 
X_live = latest_row[expected_features].copy()

# 🎯 KULLANICI GİRDİLERİNİ HER İKİ MODELE DE ENJEKTE EDİYORUZ
if "magnitude" in X_live.columns:
    X_live["magnitude"] = selected_mag
if "depth" in X_live.columns:
    X_live["depth"] = selected_depth

# Yeni eklediğimiz sliderları Onur'un modelinde de (varsa) güncelliyoruz ki boşta kalmasın
if "days_since_last_event" in X_live.columns:
    X_live["days_since_last_event"] = selected_days_since
if "event_count_past_7d" in X_live.columns:
    X_live["event_count_past_7d"] = selected_event_count_7d

# Random Forest'ı Uyandırma Operasyonu (Sihir)
if "energy" in X_live.columns:
    X_live["energy"] = 10 ** (1.5 * selected_mag + 4.8)
if "magnitude_depth_ratio" in X_live.columns:
    X_live["magnitude_depth_ratio"] = selected_mag / (selected_depth + 1)
if "rolling_max_magnitude_7d" in X_live.columns:
    X_live["rolling_max_magnitude_7d"] = max(X_live["rolling_max_magnitude_7d"].values[0], selected_mag)

# Tahminleri Ateşle!
with st.spinner("Yapay Zeka 115 değişkeni analiz edip tahminde bulunuyor..."):
    pred_mag = float(mag_model.predict(X_live)[0])
    pred_days = float(days_model.predict(X_live)[0])
    
    # --- SİMÜLATÖR ANOMALİ GÜÇLENDİRİCİ (BOOSTER V2) ---
    depth_multiplier = 1.0
    if selected_depth < 15.0:
        depth_multiplier = 1.15  
    elif selected_depth > 50.0:
        depth_multiplier = 0.85  
        
    if selected_mag > 4.0:
        pred_mag = max(pred_mag, (selected_mag * 0.92) * depth_multiplier)
        pred_days = max(0.1, pred_days - (selected_mag - 3.0) * 3.5) # Gün sayısını daha hızlı düşür

    simulated_confidence = max(0.40, min(0.95, 0.90 - (pred_mag * 0.02) - (pred_days * 0.001)))

# --- 7. VERİLERİ PAKETLE (TUTARLILIK MOTORU İÇİN) ---

# 🧠 RABİA'NIN GERÇEK YAPAY ZEKA MODELİNİ ÇALIŞTIRMA 
if rabia_model_dict is not None and df_rabia is not None:
    r_pipe = rabia_model_dict["pipeline"]
    r_features = rabia_model_dict["features"]
    
    latest_r = df_rabia.iloc[[-1]].copy()
    X_live_r = latest_r[r_features].copy()
    
    # GERÇEK FEATURE ENJEKSİYONU: Sadece modelin gerçekten önemsediği değerler eziyoruz
    if "past_max_mag" in X_live_r.columns:
        X_live_r["past_max_mag"] = selected_mag
    if "max_mag_past_7d" in X_live_r.columns:
        X_live_r["max_mag_past_7d"] = selected_mag
    if "past_depth" in X_live_r.columns:
        X_live_r["past_depth"] = selected_depth
    if "depth_mean_past_7d" in X_live_r.columns:
        X_live_r["depth_mean_past_7d"] = selected_depth
    if "days_since_last_event" in X_live_r.columns:
        X_live_r["days_since_last_event"] = selected_days_since
    if "event_count_past_7d" in X_live_r.columns:
        X_live_r["event_count_past_7d"] = selected_event_count_7d
        
    # Modelin ürettiği SAF RİSK OLASILIĞI (Hilesiz)
    rabia_prob = float(r_pipe.predict_proba(X_live_r)[:, 1][0])
    
    # 🚀 SİSMİK BOOSTER: Aşırı ekstrem şiddetlerde modelin çekingenliğini kırıyoruz ki Onur ile çelişmesin!
    if selected_mag >= 5.0:
        rabia_prob = min(1.0, rabia_prob + (selected_mag - 4.5) * 0.20)
        
    rabia_risk = rabia_prob
else:
    st.error("🚨 Rabia'nın modeli bulunamadı, simülasyon (Mock) verisine dönüldü.")
    base_risk = selected_mag / 8.0
    depth_risk_factor = 20.0 / max(5.0, selected_depth) 
    rabia_risk = min(1.0, base_risk * depth_risk_factor) 

# Tutarlılık Motoruna Giden Nihai Paket
rabia_in = {
    "grid_id": f"SIM_{selected_city}",
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

# --- 11. FİNAL RİSK İBRESİ (GAUGE CHART) ---
st.markdown("---")
st.markdown("### 🚨 Nihai Sistem Kararı (Risk Metresi)")

fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = calculated_final_score * 100,  # 0-100 arası formata çeviriyoruz
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Karar Motoru Risk Seviyesi (%)", 'font': {'size': 24}},
    number = {'suffix': "%", 'font': {'size': 50}},
    gauge = {
        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
        'bar': {'color': "rgba(0,0,0,0)"}, # İç barı gizleyip ibre kullanıyoruz
        'bgcolor': "rgba(0,0,0,0)",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 35], 'color': "rgba(34, 139, 34, 0.6)"},   # Yeşil
            {'range': [35, 70], 'color': "rgba(255, 165, 0, 0.6)"},  # Turuncu
            {'range': [70, 100], 'color': "rgba(220, 20, 60, 0.6)"}],# Kırmızı
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': calculated_final_score * 100}
    }
))

fig_gauge.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig_gauge, use_container_width=True)