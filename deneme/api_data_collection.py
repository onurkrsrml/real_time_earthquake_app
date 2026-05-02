import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

FILE_NAME = "data/earthquakes_1600_to_2026.csv"
MIN_MAGNITUDE_MODERN = 0.1
MIN_MAGNITUDE_HISTORICAL = 5.5
GAP_THRESHOLD_DAYS = 2

def fetch_usgs_paginated(start_date, end_date, min_magnitude, query_name="Sorgu"):

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    all_earthquakes = []
    current_end = end_date
    max_retries = 5
    retry_delay = 5

    print(f"\n--- {query_name} Deprem Verisi Toplama Başlıyor ---")

    while True:
        params = {
            "format": "geojson",
            "starttime": start_date,
            "endtime": current_end,
            "minmagnitude": min_magnitude,
            "orderby": "time",
            "limit": 20000
        }

        retries = 0
        response = None
        while retries < max_retries:
            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    break
                elif response.status_code in [429, 503, 504]:
                    print(
                        f"Hata {response.status_code}: Sunucu meşgul. {retry_delay}sn sonra tekrar... ({retries + 1}/{max_retries})")
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

def fetch_usgs_chunked(start_date, end_date, min_magnitude, query_name="Sorgu"):
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

        df_chunk = fetch_usgs_paginated(s_str, e_str, min_magnitude, query_name=f"{query_name} ({current_start.year})")
        if not df_chunk.empty:
            all_dfs.append(df_chunk)
        current_start = current_end

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def build_initial_dataset():
    print("\n=== SIFIRDAN VERİ TOPLAMA BAŞLATILDI ===")

    df_hist = fetch_usgs_paginated("1600-01-01T00:00:00", "1899-12-31T23:59:59", MIN_MAGNITUDE_HISTORICAL, "TARİHSEL")

    # Default olarak 1 Nisan 2026 baz alındı ancak ihtiyaç halinde anlık tarih ve zaman seçilebilir.
    # now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    now_str = "2026-04-01T00:00:00"
    df_modern = fetch_usgs_chunked("1900-01-01T00:00:00", now_str, MIN_MAGNITUDE_MODERN, "MODERN")

    df_final = pd.concat([df_hist, df_modern], ignore_index=True)
    df_final.sort_values(by="time", inplace=True)
    df_final.drop_duplicates(subset=["id"], inplace=True)

    df_final.to_csv(FILE_NAME, index=False)
    print(f"\nİşlem Tamam! {len(df_final)} kayıt kaydedildi.")
    return df_final


def find_and_fill_gaps(csv_path):
    if not os.path.exists(csv_path):
        return build_initial_dataset()

    print(f"\n--- {csv_path} kontrol ediliyor... ---")
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values(by='time').reset_index(drop=True)

    df['time_diff'] = df['time'].diff()
    gaps = df[df['time_diff'] > pd.Timedelta(days=GAP_THRESHOLD_DAYS)]

    if gaps.empty:
        print("Boşluk bulunamadı. Veri seti tam görünüyor.")
        return df

    print(f"DİKKAT: {len(gaps)} adet zaman boşluğu bulundu. Yamalar yapılıyor...")
    new_data_frames = []

    for index, row in gaps.iterrows():
        end_gap = row['time']
        start_gap = df.iloc[index - 1]['time']

        mag = MIN_MAGNITUDE_MODERN if start_gap.year >= 1900 else MIN_MAGNITUDE_HISTORICAL

        s_str = start_gap.strftime("%Y-%m-%dT%H:%M:%S")
        e_str = end_gap.strftime("%Y-%m-%dT%H:%M:%S")

        print(f"Eksik Aralığı Dolduruluyor: {s_str} -> {e_str}")
        df_missing = fetch_usgs_paginated(s_str, e_str, min_magnitude=mag, query_name="YAMA")

        if not df_missing.empty:
            new_data_frames.append(df_missing)

    if new_data_frames:
        df_combined = pd.concat([df] + new_data_frames, ignore_index=True)
        df_combined = df_combined.drop(columns=['time_diff'], errors='ignore')
        df_combined.sort_values(by='time', inplace=True)
        df_combined.drop_duplicates(subset=['id'], inplace=True)

        df_combined.to_csv(csv_path, index=False)
        print(f"Veri seti güncellendi! Yeni satır sayısı: {len(df_combined)}")
        return df_combined

    return df


if __name__ == "__main__":
    find_and_fill_gaps(FILE_NAME)