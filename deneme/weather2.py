import pandas as pd
import requests
import numpy as np
import time  # Bekleme (sleep) süreleri için eklendi
from tqdm import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

import warnings
warnings.simplefilter(action="ignore")

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    # print("##################### Quantiles #####################")
    # print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# Hava durumu WMO kodlarını Türkçe metinlere çeviren sözlük
wmo_codes = {
    0: 'Açık / Güneşli',
    1: 'Çoğunlukla Açık', 2: 'Parçalı Bulutlu', 3: 'Kapalı / Bulutlu',
    45: 'Sisli', 48: 'Kırağılı Sis',
    51: 'Hafif Çiseleyen Yağmur', 53: 'Orta Çiseleyen Yağmur', 55: 'Yoğun Çiseleyen Yağmur',
    61: 'Hafif Yağmurlu', 63: 'Orta Şiddetli Yağmurlu', 65: 'Şiddetli Yağmurlu',
    71: 'Hafif Karlı', 73: 'Orta Şiddetli Karlı', 75: 'Yoğun Karlı', 77: 'Kar Taneleri',
    80: 'Hafif Sağanak Yağışlı', 81: 'Orta Sağanak Yağışlı', 82: 'Şiddetli Sağanak Yağışlı',
    95: 'Gök Gürültülü Fırtına', 96: 'Hafif Dolulu Fırtına', 99: 'Şiddetli Dolulu Fırtına'
}


def get_weather_data(df, lat_col='latitude', lon_col='longitude', time_col='time'):
    # Bağlantı kopmalarını engellemek ve hızı artırmak için Session başlatıyoruz
    session = requests.Session()

    # 1. Zaman İşlemleri ve 1940 Öncesini Filtreleme
    df[time_col] = pd.to_datetime(df[time_col])

    # ERA5 1940'tan başladığı için veri setini filtrele
    df_filtered = df[df[time_col].dt.year >= 1940].copy()

    # Sadece tarihi ve saati yeni kolonlara al
    df_filtered['date'] = df_filtered[time_col].dt.strftime('%Y-%m-%d')
    df_filtered['hour'] = df_filtered[time_col].dt.hour

    # 2. MEKANSAL YUVARLAMA (Sorgu sayısını dramatik şekilde düşürür)
    # Koordinatları 1 ondalığa (~11 km) yuvarlıyoruz
    df_filtered['lat_round'] = df_filtered[lat_col].round(1)
    df_filtered['lon_round'] = df_filtered[lon_col].round(1)

    # 3. Sadece Benzersiz Sorguları Çıkar
    unique_reqs = df_filtered[['date', 'lat_round', 'lon_round']].drop_duplicates().reset_index(drop=True)

    print(f"Orijinal Satır Sayısı (1940 Sonrası): {len(df_filtered)}")
    print(f"API'ye Atılacak Optimize Sorgu Sayısı: {len(unique_reqs)}\n")

    weather_results = {}
    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    unique_dates = unique_reqs['date'].unique()

    # Chunk boyutunu sunucuyu yormamak için 15'e düşürdük
    chunk_size = 15

    # 4. API'den Verileri Çek
    for date in tqdm(unique_dates, desc="Hava Durumu Verisi Çekiliyor"):
        # O gün için benzersiz koordinatları al
        daily_data = unique_reqs[unique_reqs['date'] == date]
        lats = daily_data['lat_round'].tolist()
        lons = daily_data['lon_round'].tolist()

        for i in range(0, len(lats), chunk_size):
            lat_chunk = lats[i:i + chunk_size]
            lon_chunk = lons[i:i + chunk_size]

            params = {
                "latitude": lat_chunk,
                "longitude": lon_chunk,
                "start_date": date,
                "end_date": date,
                "hourly": "temperature_2m,weathercode",
                "timezone": "Europe/Istanbul"
            }

            max_deneme = 3
            for attempt in range(max_deneme):
                try:
                    # Timeout süresini 15 saniye yaptık
                    response = session.get(API_URL, params=params, timeout=15)
                    response.raise_for_status()
                    data = response.json()

                    if isinstance(data, dict):
                        data = [data]

                    # Gelen veriyi geçici sözlüğe (dictionary) işle
                    for j, loc_data in enumerate(data):
                        key = f"{date}_{lat_chunk[j]}_{lon_chunk[j]}"
                        weather_results[key] = loc_data['hourly']

                    break  # Başarılı olursa deneme döngüsünden çık

                except requests.exceptions.RequestException as e:
                    time.sleep(2)  # Hata veya timeout olursa 2 sn dinlen, tekrar dene
            else:
                print(f"\n{date} koordinat bloğu atlandı (Bağlantı kurulamadı).")

            time.sleep(0.2)  # Sunucuyu boğmamak için istekler arası kısa bir mola

    # 5. Çekilen Verileri Orijinal Veri Setine Haritala
    print("\nVeriler orijinal veri setine eşleştiriliyor...")

    def map_weather(row):
        key = f"{row['date']}_{row['lat_round']}_{row['lon_round']}"
        if key in weather_results:
            hour = row['hour']
            temp = weather_results[key]['temperature_2m'][hour]
            code = weather_results[key]['weathercode'][hour]
            return pd.Series([temp, code])
        return pd.Series([np.nan, np.nan])

    # Yeni kolonları uygula
    df_filtered[['Sicaklik', 'Hava_Durumu_Kodu']] = df_filtered.apply(map_weather, axis=1)

    # WMO kodlarını okunabilir metne çevir
    df_filtered['Hava_Durumu'] = df_filtered['Hava_Durumu_Kodu'].map(wmo_codes).fillna('Bilinmiyor')

    # Geçici hesaplama sütunlarını temizle
    df_filtered = df_filtered.drop(columns=['lat_round', 'lon_round'])

    print("İşlem başarıyla tamamlandı!")
    return df_filtered


# --- ANA VERI ÇEKME İŞLEMİ (İLK ÇALIŞTIRMA İÇİN) ---
# Not: Bu bölüm, 'hava_durumlu_depremler.csv' dosyasını sıfırdan oluşturmak için kullanılır.
# Eğer sadece eksik verileri tamamlamak istiyorsanız, bu bölümü çalıştırmanıza gerek yoktur.
#
# print("--- Veri Seti İlk Kez Oluşturuluyor ---")
# df = pd.read_csv("data/turkey_earthquakes_1900_2026.csv")
# weather_added_df = get_weather_data(df, lat_col='latitude', lon_col='longitude', time_col='time')
# weather_added_df.to_csv("hava_durumlu_depremler.csv", index=False)
# print("--- 'hava_durumlu_depremler.csv' dosyası başarıyla oluşturuldu ---")


# --- EKSIK VERILERI TAMAMLAMA KODU ---

print("\n--- Eksik Veri Tamamlama Süreci Başlatılıyor ---")

try:
    # 1. Mevcut veriyi oku
    df_mevcut = pd.read_csv("hava_durumlu_depremler.csv")
    print(f"Mevcut dosya okundu: {df_mevcut.shape[0]} satır.")

    # 'time' sütununu tekrar datetime formatına çevir, çünkü CSV'den okununca string olur.
    df_mevcut['time'] = pd.to_datetime(df_mevcut['time'])

    # 2. Eksik veriye sahip satırları bul
    df_eksik = df_mevcut[df_mevcut['Sicaklik'].isnull()].copy()
    print(f"Eksik 'Sicaklik' verisi olan {len(df_eksik)} satır bulundu.")

    if not df_eksik.empty:
        # 3. Sadece eksik veriler için hava durumu fonksiyonunu tekrar çalıştır
        print("\nEksik veriler için API'den yeniden veri çekiliyor...")
        df_doldurulmus = get_weather_data(df_eksik, lat_col='latitude', lon_col='longitude', time_col='time')

        if not df_doldurulmus.empty and 'Sicaklik' in df_doldurulmus.columns:
            newly_filled_count = df_doldurulmus['Sicaklik'].notna().sum()
            print(f"\n{newly_filled_count} satır için yeni veri çekildi. Ana tablo güncelleniyor...")

            if newly_filled_count > 0:
                # Doldurulmuş dataframe'in index'ini, eksik olanların orijinal index'i ile aynı yap.
                df_doldurulmus.set_index(df_eksik.index, inplace=True)

                # df_mevcut'u yeni verilerle güncelle. Sadece ilgili sütunlar güncellenir.
                df_mevcut.update(df_doldurulmus)

                # 4. Güncellenmiş veriyi kaydet
                df_mevcut.to_csv("hava_durumlu_depremler.csv", index=False)
                print("\nEksik veriler tamamlandı ve 'hava_durumlu_depremler.csv' dosyası güncellendi.")

                # Kontrol
                eksik_kalan = df_mevcut['Sicaklik'].isnull().sum()
                print(f"Güncelleme sonrası kalan eksik 'Sicaklik' sayısı: {eksik_kalan}")
            else:
                print("API'den yeni veri çekilemedi, dosya güncellenmedi.")
        else:
            print("Eksik veriler için API'den veri alınamadı.")
    else:
        print("Dosyada eksik 'Sicaklik' verisi bulunmuyor. İşlem yapılmadı.")

except FileNotFoundError:
    print("'hava_durumlu_depremler.csv' dosyası bulunamadı. Lütfen önce ana betiği çalıştırın.")
