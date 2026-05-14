
# ============================================================
# DeepFault Streamlit Web App - 81 İl Destekli
# Sunum/Demo Arayüzü
# ============================================================

from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# AYARLAR
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

PREDICTIONS_PATH = OUTPUT_DIR / "05_test_predictions.csv"
RAW_DATA_PATH = BASE_DIR / "depremler_hava_nasa.csv"

APP_TITLE = "DeepFault | Türkiye 81 İl Deprem Risk Skoru"

# İl merkezi etrafında risk alınacak yarıçap.
# Sunum/demo için 75 km dengeli bir seçimdir.
DEFAULT_RADIUS_KM = 75


# ============================================================
# 81 İL MERKEZ KOORDİNATLARI
# Yaklaşık il merkezi koordinatlarıdır.
# Model zaten grid-cell tabanlı olduğu için şehir poligonu değil,
# şehir merkezi + yarıçap agregasyonu kullanılır.
# ============================================================

TURKEY_PROVINCES = {
    "Adana": (37.0000, 35.3213),
    "Adıyaman": (37.7648, 38.2786),
    "Afyonkarahisar": (38.7569, 30.5387),
    "Ağrı": (39.7191, 43.0503),
    "Amasya": (40.6533, 35.8331),
    "Ankara": (39.9334, 32.8597),
    "Antalya": (36.8969, 30.7133),
    "Artvin": (41.1828, 41.8183),
    "Aydın": (37.8450, 27.8396),
    "Balıkesir": (39.6484, 27.8826),
    "Bilecik": (40.1500, 29.9833),
    "Bingöl": (38.8847, 40.4939),
    "Bitlis": (38.3938, 42.1232),
    "Bolu": (40.5760, 31.5788),
    "Burdur": (37.7203, 30.2908),
    "Bursa": (40.1828, 29.0663),
    "Çanakkale": (40.1553, 26.4142),
    "Çankırı": (40.6013, 33.6134),
    "Çorum": (40.5506, 34.9556),
    "Denizli": (37.7830, 29.0963),
    "Diyarbakır": (37.9144, 40.2306),
    "Edirne": (41.6771, 26.5557),
    "Elazığ": (38.6748, 39.2225),
    "Erzincan": (39.7500, 39.5000),
    "Erzurum": (39.9000, 41.2700),
    "Eskişehir": (39.7767, 30.5206),
    "Gaziantep": (37.0662, 37.3833),
    "Giresun": (40.9128, 38.3895),
    "Gümüşhane": (40.4603, 39.4814),
    "Hakkari": (37.5744, 43.7408),
    "Hatay": (36.2021, 36.1600),
    "Isparta": (37.7648, 30.5566),
    "Mersin": (36.8121, 34.6415),
    "İstanbul": (41.0082, 28.9784),
    "İzmir": (38.4237, 27.1428),
    "Kars": (40.6013, 43.0975),
    "Kastamonu": (41.3887, 33.7827),
    "Kayseri": (38.7205, 35.4826),
    "Kırklareli": (41.7351, 27.2252),
    "Kırşehir": (39.1458, 34.1606),
    "Kocaeli": (40.8533, 29.8815),
    "Konya": (37.8714, 32.4846),
    "Kütahya": (39.4167, 29.9833),
    "Malatya": (38.3552, 38.3095),
    "Manisa": (38.6191, 27.4289),
    "Kahramanmaraş": (37.5753, 36.9228),
    "Mardin": (37.3212, 40.7245),
    "Muğla": (37.2153, 28.3636),
    "Muş": (38.9462, 41.7539),
    "Nevşehir": (38.6244, 34.7239),
    "Niğde": (37.9667, 34.6833),
    "Ordu": (40.9862, 37.8797),
    "Rize": (41.0201, 40.5234),
    "Sakarya": (40.7569, 30.3781),
    "Samsun": (41.2867, 36.3300),
    "Siirt": (37.9333, 41.9500),
    "Sinop": (42.0264, 35.1551),
    "Sivas": (39.7477, 37.0179),
    "Tekirdağ": (40.9780, 27.5110),
    "Tokat": (40.3167, 36.5500),
    "Trabzon": (41.0015, 39.7178),
    "Tunceli": (39.1062, 39.5483),
    "Şanlıurfa": (37.1674, 38.7955),
    "Uşak": (38.6823, 29.4082),
    "Van": (38.4891, 43.4089),
    "Yozgat": (39.8181, 34.8147),
    "Zonguldak": (41.4564, 31.7987),
    "Aksaray": (38.3687, 34.0370),
    "Bayburt": (40.2552, 40.2249),
    "Karaman": (37.1811, 33.2150),
    "Kırıkkale": (39.8468, 33.5153),
    "Batman": (37.8812, 41.1351),
    "Şırnak": (37.5164, 42.4611),
    "Bartın": (41.5811, 32.4610),
    "Ardahan": (41.1105, 42.7022),
    "Iğdır": (39.9237, 44.0450),
    "Yalova": (40.6500, 29.2667),
    "Karabük": (41.2061, 32.6204),
    "Kilis": (36.7184, 37.1212),
    "Osmaniye": (37.0742, 36.2478),
    "Düzce": (40.8438, 31.1565),
}

# Kullanıcı Türkçe karakter yazmasa da yakalansın.
ALIASES = {
    "istanbul": "İstanbul",
    "izmir": "İzmir",
    "adiyaman": "Adıyaman",
    "agri": "Ağrı",
    "aydin": "Aydın",
    "balikesir": "Balıkesir",
    "bilecik": "Bilecik",
    "bingol": "Bingöl",
    "bitlis": "Bitlis",
    "canakkale": "Çanakkale",
    "cankiri": "Çankırı",
    "corum": "Çorum",
    "diyarbakir": "Diyarbakır",
    "edirne": "Edirne",
    "elazig": "Elazığ",
    "erzincan": "Erzincan",
    "eskisehir": "Eskişehir",
    "gumushane": "Gümüşhane",
    "igdir": "Iğdır",
    "kirklareli": "Kırklareli",
    "kirsehir": "Kırşehir",
    "kutahya": "Kütahya",
    "maras": "Kahramanmaraş",
    "kahramanmaras": "Kahramanmaraş",
    "mugla": "Muğla",
    "mus": "Muş",
    "nevsehir": "Nevşehir",
    "nigde": "Niğde",
    "sanliurfa": "Şanlıurfa",
    "urfa": "Şanlıurfa",
    "sirnak": "Şırnak",
    "tekirdag": "Tekirdağ",
    "usak": "Uşak",
    "zonguldak": "Zonguldak",
    "duzce": "Düzce",
    "mersin": "Mersin",
    "icel": "Mersin",
}


# ============================================================
# SAYFA AYARI
# ============================================================

st.set_page_config(
    page_title="DeepFault 81 İl Risk Skoru",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("ı", "i").replace("İ", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    return text


@st.cache_data
def load_predictions():
    if not PREDICTIONS_PATH.exists():
        return None

    df = pd.read_csv(PREDICTIONS_PATH)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    rename_map = {
        "risk_probability": "risk_score",
        "probability": "risk_score",
        "pred_prob": "risk_score",
        "y_prob": "risk_score",
        "lat": "latitude",
        "lon": "longitude",
        "lng": "longitude",
        "date": "time"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # risk_score yoksa olası kolonları bul
    if "risk_score" not in df.columns:
        candidates = [c for c in df.columns if "risk" in c or "prob" in c]
        if candidates:
            df = df.rename(columns={candidates[0]: "risk_score"})

    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vektörel haversine mesafesi.
    """
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def canonical_city_name(user_text):
    """
    Soru metninden veya selectbox değerinden 81 il adını yakalar.
    """
    if user_text is None:
        return None

    normalized = normalize_text(user_text)

    # Direkt alias
    for alias, canonical in ALIASES.items():
        if alias in normalized:
            return canonical

    # İl adları
    for city in TURKEY_PROVINCES.keys():
        if normalize_text(city) in normalized:
            return city

    return None


def risk_level(score):
    if pd.isna(score):
        return "Bilinmiyor", "Bu şehir için yeterli tahmin noktası bulunamadı.", "gray"

    if score < 0.30:
        return "Düşük Risk", "Model kısa vadeli bölgesel riski düşük görüyor.", "green"
    elif score < 0.60:
        return "Orta Risk", "Model bölgede dikkat edilmesi gereken orta seviye risk sinyali görüyor.", "orange"
    else:
        return "Yüksek Risk", "Model bölgede yüksek risk sinyali görüyor. Bu kesin deprem tahmini değildir.", "red"


def get_province_risk(pred_df, province, radius_km=DEFAULT_RADIUS_KM):
    """
    İl merkezi koordinatına göre, belirli yarıçap içindeki grid tahminlerini toplar.
    Eğer yarıçap içinde veri yoksa en yakın N tahmin noktasını fallback olarak kullanır.
    """
    if pred_df is None or pred_df.empty:
        return None

    required = {"latitude", "longitude", "risk_score"}
    if not required.issubset(set(pred_df.columns)):
        return None

    if province not in TURKEY_PROVINCES:
        return None

    center_lat, center_lon = TURKEY_PROVINCES[province]

    df = pred_df.dropna(subset=["latitude", "longitude", "risk_score"]).copy()

    distances = haversine_km(
        center_lat,
        center_lon,
        df["latitude"].to_numpy(),
        df["longitude"].to_numpy()
    )

    df["distance_km"] = distances

    local = df[df["distance_km"] <= radius_km].copy()
    fallback_used = False

    # Eğer o il çevresinde grid yoksa en yakın 25 noktayı al.
    if local.empty:
        local = df.sort_values("distance_km").head(25).copy()
        fallback_used = True

    if local.empty:
        return None

    # Son tarih varsa son dönemi ağırlıklı göstermek için son 90 günlük subset de hesapla.
    recent = local
    if "time" in local.columns and local["time"].notna().any():
        max_time = local["time"].max()
        recent = local[local["time"] >= max_time - pd.Timedelta(days=90)].copy()
        if recent.empty:
            recent = local

    result = {
        "province": province,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_km": radius_km,
        "fallback_used": fallback_used,
        "n_points": len(local),
        "n_recent_points": len(recent),
        "mean_risk": float(recent["risk_score"].mean()),
        "median_risk": float(recent["risk_score"].median()),
        "max_risk": float(recent["risk_score"].max()),
        "p75_risk": float(recent["risk_score"].quantile(0.75)),
        "latest_time": recent["time"].max() if "time" in recent.columns else None,
        "data": recent
    }
    return result


def show_risk_card(title, score):
    level, comment, color = risk_level(score)

    st.markdown(
        f"""
        <div style="
            border-radius: 18px;
            padding: 22px;
            border: 1px solid #e5e7eb;
            background-color: #ffffff;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            min-height: 210px;
        ">
            <h4 style="margin-bottom: 8px;">{title}</h4>
            <h1 style="color:{color}; margin-top: 0;">{score:.3f}</h1>
            <h3 style="color:{color};">{level}</h3>
            <p style="font-size:15px;">{comment}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def build_all_province_table(pred_df, radius_km):
    rows = []
    for province in TURKEY_PROVINCES:
        res = get_province_risk(pred_df, province, radius_km=radius_km)
        if res is None:
            rows.append({
                "il": province,
                "ortalama_risk": np.nan,
                "medyan_risk": np.nan,
                "maksimum_risk": np.nan,
                "risk_seviyesi": "Bilinmiyor",
                "nokta_sayısı": 0,
            })
        else:
            level, _, _ = risk_level(res["mean_risk"])
            rows.append({
                "il": province,
                "ortalama_risk": res["mean_risk"],
                "medyan_risk": res["median_risk"],
                "maksimum_risk": res["max_risk"],
                "risk_seviyesi": level,
                "nokta_sayısı": res["n_points"],
            })
    return pd.DataFrame(rows).sort_values("ortalama_risk", ascending=False)


# ============================================================
# APP
# ============================================================

st.title(APP_TITLE)

st.markdown(
    """
Bu demo, DeepFault modelinin ürettiği tahminleri Türkiye'nin **81 ili** için sorgulanabilir hale getirir.

> Sistem kesin deprem tahmini yapmaz. Belirli bir il çevresindeki grid-cell tahminlerini kullanarak
> **önümüzdeki 7 gün için bölgesel risk skoru** gösterir.
"""
)

pred_df = load_predictions()

if pred_df is None:
    st.error(
        "outputs/05_test_predictions.csv bulunamadı. "
        "Önce V4 model pipeline'ını çalıştırıp outputs klasörünü oluşturmalısın."
    )
    st.stop()

required_cols = {"latitude", "longitude", "risk_score"}
if not required_cols.issubset(set(pred_df.columns)):
    st.error(
        f"Tahmin dosyasında gerekli kolonlar eksik. Gerekli: {required_cols}. "
        f"Mevcut kolonlar: {list(pred_df.columns)}"
    )
    st.stop()


# Sidebar
st.sidebar.header("Kontrol Paneli")

radius_km = st.sidebar.slider(
    "İl çevresi yarıçapı (km)",
    min_value=25,
    max_value=150,
    value=DEFAULT_RADIUS_KM,
    step=25
)

mode = st.sidebar.radio(
    "Sorgu tipi",
    ["İl seç", "Soru yaz", "81 il tablosu"]
)

selected_province = None

province_list = sorted(TURKEY_PROVINCES.keys())

if mode == "İl seç":
    default_city = "İstanbul"
    default_index = province_list.index(default_city) if default_city in province_list else 0
    selected_province = st.sidebar.selectbox("İl seç", province_list, index=default_index)

elif mode == "Soru yaz":
    question = st.text_input(
        "Soru yaz:",
        placeholder="Örnek: İstanbul için önümüzdeki 7 gün deprem risk skoru nedir?"
    )
    selected_province = canonical_city_name(question)

    if question and selected_province is None:
        st.warning("Soruda il adı yakalayamadım. Örnek: 'İstanbul için risk nedir?'")

else:
    selected_province = None


# Genel özet
st.subheader("Genel Model Çıktı Özeti")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Tahmin Satırı", f"{len(pred_df):,}")

with c2:
    st.metric("Ortalama Risk", f"{pred_df['risk_score'].mean():.3f}")

with c3:
    st.metric("Maksimum Risk", f"{pred_df['risk_score'].max():.3f}")

with c4:
    if "prediction" in pred_df.columns:
        st.metric("Alarm Oranı", f"{pred_df['prediction'].mean():.2%}")
    else:
        st.metric("Alarm Oranı", "Yok")


# 81 il tablo modu
if mode == "81 il tablosu":
    st.divider()
    st.subheader("81 İl Risk Sıralaması")

    all_table = build_all_province_table(pred_df, radius_km)
    st.dataframe(all_table, use_container_width=True)

    st.download_button(
        "81 il risk tablosunu indir",
        data=all_table.to_csv(index=False).encode("utf-8-sig"),
        file_name="deepfault_81_il_risk_tablosu.csv",
        mime="text/csv"
    )

    top10 = all_table.head(10).set_index("il")[["ortalama_risk"]]
    st.markdown("### En yüksek ortalama risk skoruna sahip ilk 10 il")
    st.bar_chart(top10)

# Tek il sonucu
if selected_province:
    st.divider()
    st.subheader(f"{selected_province} için 7 Günlük Bölgesel Risk Skoru")

    result = get_province_risk(pred_df, selected_province, radius_km=radius_km)

    if result is None:
        st.error(f"{selected_province} için risk skoru üretilemedi.")
    else:
        if result["fallback_used"]:
            st.warning(
                f"{selected_province} için {radius_km} km içinde tahmin noktası bulunamadı. "
                "Demo amacıyla en yakın tahmin noktaları kullanıldı."
            )

        a, b, c = st.columns(3)

        with a:
            show_risk_card("Ortalama Risk", result["mean_risk"])

        with b:
            show_risk_card("Medyan Risk", result["median_risk"])

        with c:
            show_risk_card("Maksimum Risk", result["max_risk"])

        level, comment, _ = risk_level(result["mean_risk"])

        st.markdown("### Model Yorumu")

        st.write(
            f"""
**{selected_province}** için modelin hesapladığı ortalama 7 günlük bölgesel risk skoru
**{result["mean_risk"]:.3f}** seviyesindedir.

Bu skor **{level}** kategorisine karşılık gelir.

Bu sonuç, il merkezinin yaklaşık **{radius_km} km** çevresindeki grid-cell tahminlerinin
agregasyonu ile hesaplanmıştır. Model; geçmiş sismik aktivite, rolling/lag değişkenleri,
bölgesel yoğunluk, meteorolojik anomaliler ve astronomik göstergelerden yararlanır.
"""
        )

        if result["latest_time"] is not None and not pd.isna(result["latest_time"]):
            st.info(f"Son tahmin zamanı: {result['latest_time']}")

        st.markdown("### İl Çevresi Tahmin Noktaları")

        local_data = result["data"].copy()

        show_cols = [c for c in ["time", "latitude", "longitude", "distance_km", "risk_score", "prediction", "target"] if c in local_data.columns]
        st.dataframe(local_data[show_cols].sort_values("risk_score", ascending=False).head(100), use_container_width=True)

        if "time" in local_data.columns:
            st.markdown("### Zaman İçinde Risk Skoru")
            chart_df = (
                local_data
                .dropna(subset=["time", "risk_score"])
                .sort_values("time")
                .groupby("time", as_index=True)["risk_score"]
                .mean()
            )
            st.line_chart(chart_df)

        st.markdown("### Harita")

        map_df = local_data[["latitude", "longitude", "risk_score"]].dropna().copy()
        st.map(
            map_df,
            latitude="latitude",
            longitude="longitude",
            size=30
        )

# Türkiye haritası
st.divider()
st.subheader("Türkiye Geneli Tahmin Noktaları")

map_all = pred_df[["latitude", "longitude", "risk_score"]].dropna().sample(
    min(len(pred_df), 5000),
    random_state=42
)
st.map(map_all, latitude="latitude", longitude="longitude", size=15)

with st.expander("Bilimsel ve metodolojik not"):
    st.markdown(
        """
- Model şehir sınırı poligonu ile değil, grid-cell tahminleriyle çalışır.
- Bu arayüzde 81 il için il merkezi + yarıçap agregasyonu yapılır.
- Skor, kesin deprem tahmini değil, 7 günlük bölgesel risk skorudur.
- Time-series feature'lar geçmişten üretilmiştir.
- Random split yerine time-based / walk-forward validation yaklaşımı kullanılmıştır.
"""
    )
# --- DIŞARIDAN GELEN İSTEKLER İÇİN API MANTIĞI ---
# Bu kısım ortak repodaki buton basıldığında çalışacak gizli motor

params = st.query_params  # Dışarıdan gelen parametreleri al

import json
import pandas as pd

# 1. Arkadaşından gelen kısıtlı veriyi al
params = st.query_params
if "latitude" in params and "longitude" in params:
    lat = float(params["latitude"])
    lon = float(params["longitude"])
    time = params.get("time", "2026-01-01")

    # 2. GİZLİ OPERASYON: Modelin beklediği tüm sütunları oluştur
    # Arkadaşın magnitude/depth bilmiyor, o yüzden burada varsayılan veya 
    # senin CSV'nden çektiğin değerleri modele veriyoruz.
    input_data = pd.DataFrame([{
        "time": time,
        "latitude": lat,
        "longitude": lon,
        "magnitude": 4.0, # Örnek sabit değer veya CSV'den eşleştirme
        "depth_km": 10.0,  # Örnek sabit değer veya CSV'den eşleştirme
        "solar_flux_f107": 150.0 # Gizli sütunlarını burada besle
    }])

    # Senin gönderdiğin o temizleme fonksiyonunu burada çağır
    # input_data = df_cleaner(input_data) 

    # 3. TAHMİN ET VE GÖNDER
    # risk_skoru = model.predict(input_data)[0]
    risk_skoru = 0.85 # Buraya kendi model sonucunu bağla
    
    st.write(json.dumps({"risk_score": float(risk_skoru)}))
    st.stop()
