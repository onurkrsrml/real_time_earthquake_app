import pandas as pd
import requests
import numpy as np
import time
import os
import csv
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

import warnings
warnings.simplefilter(action="ignore")

# Hava durumu WMO kodlarını Türkçe metinlere çeviren sözlük
# wmo_codes = {
#     0: 'Açık / Güneşli', 1: 'Çoğunlukla Açık', 2: 'Parçalı Bulutlu', 3: 'Kapalı / Bulutlu',
#     45: 'Sisli', 48: 'Kırağılı Sis', 51: 'Hafif Çiseleyen Yağmur', 53: 'Orta Çiseleyen Yağmur',
#     55: 'Yoğun Çiseleyen Yağmur', 61: 'Hafif Yağmurlu', 63: 'Orta Şiddetli Yağmurlu', 65: 'Şiddetli Yağmurlu',
#     71: 'Hafif Karlı', 73: 'Orta Şiddetli Karlı', 75: 'Yoğun Karlı', 77: 'Kar Taneleri',
#     80: 'Hafif Sağanak Yağışlı', 81: 'Orta Sağanak Yağışlı', 82: 'Şiddetli Sağanak Yağışlı',
#     95: 'Gök Gürültülü Fırtına', 96: 'Hafif Dolulu Fırtına', 99: 'Şiddetli Dolulu Fırtına'
# }

wmo_codes = {
    0: 'Clear sky',
    1: 'Mainly clear',
    2: 'Partly cloudy',
    3: 'Overcast',
    45: 'Fog',
    48: 'Depositing rime fog',
    51: 'Light drizzle',
    53: 'Moderate drizzle',
    55: 'Dense drizzle',
    61: 'Slight rain',
    63: 'Moderate rain',
    65: 'Heavy rain',
    71: 'Slight snow fall',
    73: 'Moderate snow fall',
    75: 'Heavy snow fall',
    77: 'Snow grains',
    80: 'Slight rain showers',
    81: 'Moderate rain showers',
    82: 'Violent rain showers',
    95: 'Thunderstorm',
    96: 'Thunderstorm with slight hail',
    99: 'Thunderstorm with heavy hail'
}

def get_weather_data(df, lat_col='latitude', lon_col='longitude', time_col='time'):
    session = requests.Session()
    df[time_col] = pd.to_datetime(df[time_col])
    df_filtered = df[df[time_col].dt.year >= 1940].copy()
    df_filtered['date'] = df_filtered[time_col].dt.strftime('%Y-%m-%d')
    df_filtered['hour'] = df_filtered[time_col].dt.hour
    df_filtered['lat_round'] = df_filtered[lat_col].round(1)
    df_filtered['lon_round'] = df_filtered[lon_col].round(1)
    unique_reqs = df_filtered[['date', 'lat_round', 'lon_round']].drop_duplicates().reset_index(drop=True)
    print(f"Open-Meteo API'ye Atılacak Optimize Sorgu Sayısı: {len(unique_reqs)}\n")
    weather_results = {}
    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    unique_dates = unique_reqs['date'].unique()
    chunk_size = 15
    for date in tqdm(unique_dates, desc="Hava Durumu Verisi Çekiliyor"):
        daily_data = unique_reqs[unique_reqs['date'] == date]
        lats = daily_data['lat_round'].tolist()
        lons = daily_data['lon_round'].tolist()
        for i in range(0, len(lats), chunk_size):
            lat_chunk = lats[i:i + chunk_size]
            lon_chunk = lons[i:i + chunk_size]
            params = {"latitude": lat_chunk, "longitude": lon_chunk, "start_date": date, "end_date": date, "hourly": "temperature_2m,weathercode", "timezone": "Europe/Istanbul"}
            for attempt in range(3):
                try:
                    response = session.get(API_URL, params=params, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                    if isinstance(data, dict): data = [data]
                    for j, loc_data in enumerate(data):
                        key = f"{date}_{lat_chunk[j]}_{lon_chunk[j]}"
                        weather_results[key] = loc_data['hourly']
                    break
                except requests.exceptions.RequestException:
                    time.sleep(2)
            else:
                print(f"\n{date} koordinat bloğu atlandı (Bağlantı kurulamadı).")
            time.sleep(0.2)
    print("\nVeriler orijinal veri setine eşleştiriliyor...")
    def map_weather(row):
        key = f"{row['date']}_{row['lat_round']}_{row['lon_round']}"
        if key in weather_results:
            hour = row['hour']
            temp = weather_results[key]['temperature_2m'][hour]
            code = weather_results[key]['weathercode'][hour]
            return pd.Series([temp, code])
        return pd.Series([np.nan, np.nan])
    df_filtered[['Sicaklik', 'Hava_Durumu_Kodu']] = df_filtered.apply(map_weather, axis=1)
    df_filtered['Hava_Durumu'] = df_filtered['Hava_Durumu_Kodu'].map(wmo_codes).fillna('Bilinmiyor')
    df_filtered = df_filtered.drop(columns=['lat_round', 'lon_round', 'date', 'hour'])
    print("İşlem başarıyla tamamlandı!")
    return df_filtered

def get_nasa_power_data(df, lat_col='latitude', lon_col='longitude', time_col='time'):
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    df[time_col] = pd.to_datetime(df[time_col])
    df_copy = df[df[time_col].dt.year >= 1981].copy()
    print(f"1981 ve sonrası için filtrelendi: {df_copy.shape[0]} satır işleme alınacak.")

    df_copy['date_str'] = df_copy[time_col].dt.strftime('%Y-%m-%d')
    df_copy['lat_round'] = df_copy[lat_col].round(1)
    df_copy['lon_round'] = df_copy[lon_col].round(1)
    
    unique_reqs = df_copy[['date_str', 'lat_round', 'lon_round']].drop_duplicates().reset_index(drop=True)
    
    # --- KALDIĞI YERDEN DEVAM ETME (CACHE) MEKANİZMASI ---
    cache_file = "nasa_api_cache.csv"
    nasa_results = {}
    
    if os.path.exists(cache_file):
        print(f"\nÖnbellek dosyası bulundu: '{cache_file}'. Kaldığı yerden devam ediliyor...")
        try:
            cache_df = pd.read_csv(cache_file)
            for _, row in cache_df.iterrows():
                key = f"{row['date_str']}_{row['lat']}_{row['lon']}"
                nasa_results[key] = {
                    'T2M': row['T2M'],
                    'RH2M': row['RH2M'],
                    'PS': row['PS']
                }
            print(f"Önbellekten {len(nasa_results)} adet önceden çekilmiş sorgu yüklendi.")
        except Exception as e:
            print(f"Önbellek okuma hatası: {e}. Dosya baştan oluşturulacak.")
            nasa_results = {}
    
    if not os.path.exists(cache_file) or len(nasa_results) == 0:
        # Cache dosyası yoksa veya hatalıysa başlıklarla oluştur
        pd.DataFrame(columns=['date_str', 'lat', 'lon', 'T2M', 'RH2M', 'PS']).to_csv(cache_file, index=False)

    # Zaten çekilmiş sorguları filtrele
    def is_cached(row):
        key = f"{row['date_str']}_{row['lat_round']}_{row['lon_round']}"
        return key not in nasa_results

    reqs_to_fetch = unique_reqs[unique_reqs.apply(is_cached, axis=1)].reset_index(drop=True)
    
    print(f"Toplam benzersiz lokasyon-tarih kombinasyonu: {len(unique_reqs)}")
    print(f"Daha önce çekilmeyen ve şimdi API'ye atılacak sorgu sayısı: {len(reqs_to_fetch)}")

    API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    # Sadece eksik kalan sorguları çek
    if not reqs_to_fetch.empty:
        # Cache dosyasını append (ekleme) modunda açıyoruz
        with open(cache_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for index, row in tqdm(reqs_to_fetch.iterrows(), total=reqs_to_fetch.shape[0], desc="NASA POWER Eksik Verileri Çekiliyor"):
                lat, lon, date_str = row['lat_round'], row['lon_round'], row['date_str']
                start_date = pd.to_datetime(date_str).strftime('%Y%m%d')
                
                params = {
                    "parameters": "T2M,RH2M,PS", 
                    "community": "AG", 
                    "longitude": lon, 
                    "latitude": lat, 
                    "start": start_date, 
                    "end": start_date, 
                    "format": "JSON"
                }
                
                max_retries = 4
                success = False
                
                for attempt in range(max_retries):
                    try:
                        response = session.get(API_URL, params=params, timeout=(10, 30))
                        
                        if response.status_code == 429:
                            time.sleep(5 * (attempt + 1))
                            continue
                        elif response.status_code >= 500:
                            time.sleep(3 * (attempt + 1))
                            continue
                            
                        response.raise_for_status()
                        data = response.json()
                        key = f"{date_str}_{lat}_{lon}"
                        
                        result_params = data['properties']['parameter']
                        t2m = result_params.get('T2M', {}).get(start_date, np.nan)
                        rh2m = result_params.get('RH2M', {}).get(start_date, np.nan)
                        ps = result_params.get('PS', {}).get(start_date, np.nan)
                        
                        # Eksik/Hatalı verileri -999'dan NaN'a çeviriyoruz
                        t2m = np.nan if t2m == -999 else t2m
                        rh2m = np.nan if rh2m == -999 else rh2m
                        ps = np.nan if ps == -999 else ps
                        
                        # RAM'deki sözlüğe kaydet
                        nasa_results[key] = {'T2M': t2m, 'RH2M': rh2m, 'PS': ps}
                        
                        # Anında DISK'e yaz (Uygulama çökse bile bu satır kaydedilmiş olur)
                        writer.writerow([date_str, lat, lon, t2m, rh2m, ps])
                        f.flush()
                        
                        success = True
                        break 
                        
                    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException, ValueError):
                        time.sleep(2 * (attempt + 1))
                    except KeyError:
                        break 
                
                if not success:
                    print(f"\nUyarı: {lat},{lon} @ {start_date} için veri çekilemedi, atlanıyor.")
                    
                time.sleep(0.35) 
        
    print("\nNASA verileri orijinal veri setine eşleştiriliyor...")
    def map_nasa_data(row):
        key = f"{row['date_str']}_{row['lat_round']}_{row['lon_round']}"
        if key in nasa_results:
            result = nasa_results[key]
            return pd.Series([result.get('T2M', np.nan), result.get('RH2M', np.nan), result.get('PS', np.nan)])
        return pd.Series([np.nan, np.nan, np.nan])
        
    df_copy[['Sicaklik_NASA', 'Nem_NASA', 'Basinc_NASA']] = df_copy.apply(map_nasa_data, axis=1)
    df_copy = df_copy.drop(columns=['date_str', 'lat_round', 'lon_round'])
    
    df_final = df.copy()
    
    for col in ['Sicaklik_NASA', 'Nem_NASA', 'Basinc_NASA']:
        if col not in df_final.columns:
            df_final[col] = np.nan
            
    df_final.update(df_copy[['Sicaklik_NASA', 'Nem_NASA', 'Basinc_NASA']])
            
    return df_final

# --- ANA VERI ÇEKME İŞLEMİ (İLK ÇALIŞTIRMA İÇİN) ---
# ...
# --- EKSIK VERILERI TAMAMLAMA KODU ---
# ...

# --- NASA POWER VERİSİ EKLEME KODU ---
print("\n--- NASA POWER Veri Ekleme Süreci Başlatılıyor ---")
try:
    df_hava_durumlu = pd.read_csv("hava_durumlu_depremler.csv")
    print(f"Mevcut dosya okundu: {df_hava_durumlu.shape[0]} satır.")
    
    df_nasa_enriched = get_nasa_power_data(df_hava_durumlu)
    
    print("\n--- Veri Seti İşleniyor: Eksik Veri Doldurma ve Sıralama ---")
    
    # DİKKAT: Veriler henüz kendi doğal indeksindeyken (satır satır eşleşiyorken) doldurma işlemi yapılıyor.
    # Bu sayede her satırdaki NaN değeri, BİREBİR AYNI SATIRDAKİ (aynı depremdeki) Sicaklik verisi ile dolar.
    
    # 1. Sicaklik_NASA'daki eksik (NaN) değerleri, önceden var olan Sicaklik sütunu ile doldur
    if 'Sicaklik' in df_nasa_enriched.columns:
        print("Sicaklik_NASA'daki eksik değerler, aynı satırdaki 'Sicaklik' verisi ile dolduruluyor...")
        df_nasa_enriched['Sicaklik_NASA'] = df_nasa_enriched['Sicaklik_NASA'].fillna(df_nasa_enriched['Sicaklik'])
        
    # 2. Nem_NASA ve Basinc_NASA'daki eksik (NaN) değerleri sütun ortalaması ile doldur
    print("Nem_NASA ve Basinc_NASA eksikleri sütunların ortalaması ile dolduruluyor...")
    if 'Nem_NASA' in df_nasa_enriched.columns:
        df_nasa_enriched['Nem_NASA'] = df_nasa_enriched['Nem_NASA'].fillna(df_nasa_enriched['Nem_NASA'].mean())
    if 'Basinc_NASA' in df_nasa_enriched.columns:
        df_nasa_enriched['Basinc_NASA'] = df_nasa_enriched['Basinc_NASA'].fillna(df_nasa_enriched['Basinc_NASA'].mean())

    # 3. İŞLEMLER BİTTİKTEN SONRA ekrana yazdırmak veya kaydetmek için tarihi yeniden eskiye sırala
    print("Veri seti tarihe göre yeniden eskiye (descending) sıralanıyor...")
    df_nasa_enriched['time'] = pd.to_datetime(df_nasa_enriched['time'])
    df_nasa_enriched = df_nasa_enriched.sort_values(by='time', ascending=True)
    
    output_filename = "depremler_hava_nasa.csv"
    df_nasa_enriched.to_csv(output_filename, index=True)
    
    print(f"\nİşlemler tamamlandı ve '{output_filename}' dosyasına kaydedildi.")
    print("\nSon DataFrame'in ilk 5 satırı (En Yeni Depremler):")
    print(df_nasa_enriched.head())
    print("\nEklenen NASA sütunlarındaki kalan eksik veri sayısı:")
    print(df_nasa_enriched[['Sicaklik_NASA', 'Nem_NASA', 'Basinc_NASA']].isnull().sum())

except FileNotFoundError:
    print("'hava_durumlu_depremler.csv' dosyası bulunamadı. Lütfen önce diğer betikleri çalıştırın.")
except Exception as e:
    print(f"Bir hata oluştu: {e}")