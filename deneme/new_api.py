import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

FILE_NAME = "data/earthquakes_turkey_1900_onwards.csv"
MIN_MAGNITUDE = 0.1

def fetch_usgs_paginated_turkey(start_date, end_date, min_magnitude, query_name="Sorgu"):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    all_earthquakes = []
    current_end = end_date
    max_retries = 5
    retry_delay = 5

    print(f"\n--- {query_name} Türkiye Deprem Verisi Toplama Başlıyor ---")

    while True:
        # Türkiye yaklaşık koordinatları (Bounding box)
        params = {
            "format": "geojson",
            "starttime": start_date,
            "endtime": current_end,
            "minmagnitude": min_magnitude,
            "orderby": "time",
            "limit": 20000,
            "minlatitude": 35.8,
            "maxlatitude": 42.1,
            "minlongitude": 25.6,
            "maxlongitude": 44.8
        }

        retries = 0
        response = None
        while retries < max_retries:
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    break
                elif response.status_code in [429, 503, 504]:
                    print(f"Hata {response.status_code}: Sunucu meşgul. {retry_delay}sn sonra tekrar... ({retries + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retries += 1
                else:
                    response.raise_for_status()
            except Exception as e:
                print(f"Bağlantı hatası: {e}. Tekrar deneniyor... ({retries + 1}/{max_retries})")
                time.sleep(retry_delay)
                retries += 1

        if not response or response.status_code != 200:
            print(f"Kritik Hata: {query_name} durduruldu.")
            break

        features = response.json().get('features', [])
        count = len(features)

        if count == 0:
            break

        print(f"Veri alındı: {current_end} öncesi {count} deprem...")

        for feature in features:
            props = feature.get('properties', {})
            coords = feature.get('geometry', {}).get('coordinates', [])
            time_ms = props.get('time')
            eq_time = datetime.fromtimestamp(time_ms / 1000.0) if time_ms else None

            all_earthquakes.append({
                "id": feature.get('id'),
                "time": eq_time,
                "magnitude": props.get('mag'),
                "place": props.get('place'),
                "longitude": coords[0] if len(coords) > 0 else None,
                "latitude": coords[1] if len(coords) > 1 else None,
                "depth_km": coords[2] if len(coords) > 2 else None,
                "source": "USGS"
            })

        if count < 20000:
            break
        else:
            oldest_time_ms = features[-1]['properties']['time']
            next_end_time_dt = datetime.fromtimestamp((oldest_time_ms - 1) / 1000.0)
            current_end = next_end_time_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

    return pd.DataFrame(all_earthquakes)

def fetch_usgs_chunked_turkey(start_date, end_date, min_magnitude, query_name="Sorgu"):
    start_dt = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")
    end_dt = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S")

    all_dfs = []
    current_start = start_dt
    chunk_size_years = 2

    while current_start < end_dt:
        current_end = current_start + timedelta(days=chunk_size_years * 365.25)
        if current_end > end_dt:
            current_end = end_dt

        s_str = current_start.strftime("%Y-%m-%dT%H:%M:%S")
        e_str = current_end.strftime("%Y-%m-%dT%H:%M:%S")

        df_chunk = fetch_usgs_paginated_turkey(s_str, e_str, min_magnitude, query_name=f"{query_name} ({current_start.year})")
        if not df_chunk.empty:
            all_dfs.append(df_chunk)
        current_start = current_end

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def build_turkey_dataset():
    print("\n=== TÜRKİYE İÇİN SIFIRDAN VERİ TOPLAMA BAŞLATILDI ===")

    # 1900 sonrasındaki bütün depremler
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    df_modern = fetch_usgs_chunked_turkey("1900-01-01T00:00:00", now_str, MIN_MAGNITUDE, "MODERN_TR")

    if not df_modern.empty:
        df_modern.sort_values(by="time", inplace=True)
        df_modern.drop_duplicates(subset=["id"], inplace=True)
        
        # data dizini yoksa oluştur
        os.makedirs(os.path.dirname(FILE_NAME), exist_ok=True)
        
        df_modern.to_csv(FILE_NAME, index=False)
        print(f"\nİşlem Tamam! {len(df_modern)} adet Türkiye depremi '{FILE_NAME}' dosyasına kaydedildi.")
    else:
        print("\nHiç veri bulunamadı!")
        
    return df_modern

if __name__ == "__main__":
    build_turkey_dataset()
