import pandas as pd
import requests
import time
import os

# Veri seti yollari
DATA_PATH = "data/turkey_earthquakes_1900_2026.csv"
OUTPUT_PATH = "data/turkey_earthquakes_1900_2026_with_weather.csv"

# Open-Meteo API WMO Hava Durumu Kodlari ve Aciklamalari
WEATHER_CODES = {
    0: "Clear sky (Gunesli/Acik)",
    1: "Mainly clear (Cogunlukla acik)", 2: "Partly cloudy (Parcali bulutlu)", 3: "Overcast (Cok bulutlu)",
    45: "Fog (Sisli)", 48: "Depositing rime fog (Kırağılı sis)",
    51: "Drizzle: Light (Hafif ciseleme)", 53: "Drizzle: Moderate (Orta ciseleme)", 55: "Drizzle: Dense (Yogun ciseleme)",
    56: "Freezing Drizzle: Light (Hafif dondurucu ciseleme)", 57: "Freezing Drizzle: Dense (Yogun dondurucu ciseleme)",
    61: "Rain: Slight (Hafif yagmur)", 63: "Rain: Moderate (Orta sidetli yagmur)", 65: "Rain: Heavy (Siddetli yagmur)",
    66: "Freezing Rain: Light (Hafif dondurucu yagmur)", 67: "Freezing Rain: Heavy (Siddetli dondurucu yagmur)",
    71: "Snow fall: Slight (Hafif kar yagisi)", 73: "Snow fall: Moderate (Orta kar yagisi)", 75: "Snow fall: Heavy (Siddetli kar yagisi)",
    77: "Snow grains (Kar taneleri)",
    80: "Rain showers: Slight (Hafif saganak yagmur)", 81: "Rain showers: Moderate (Orta saganak yagmur)", 82: "Rain showers: Violent (Siddetli saganak yagmur)",
    85: "Snow showers slight (Hafif saganak kar)", 86: "Snow showers heavy (Siddetli saganak kar)",
    95: "Thunderstorm: Slight or moderate (Hafif veya orta gok gurultulu firtina)",
    96: "Thunderstorm with slight hail (Hafif dolulu gok gurultulu firtina)", 99: "Thunderstorm with heavy hail (Siddetli dolulu firtina)",
}

def get_weather(lat, lon, date_str):
    """
    Open-Meteo Historical Weather API (Archive) kullanarak hava durumu bilgisini ceker.
    API ucretsizdir ancak gunde 10.000 istek siniri vardir. 1940 yili oncesi icin veri saglamaz.
    """
    # Sadece tarih kismini al (saati at, YYYY-MM-DD)
    date_only = str(date_str).split(" ")[0]
    
    try:
        year = int(date_only.split("-")[0])
        # Open-Meteo Archive API sadece 1940 ve sonrasi icin veri sagliyor
        if year < 1940:
            return None, "No Data (Pre-1940)"
    except Exception:
        return None, "Invalid Date"

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_only,
        "end_date": date_only,
        "daily": "temperature_2m_mean,weathercode",
        "timezone": "auto"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "daily" in data and len(data["daily"]["temperature_2m_mean"]) > 0:
                temp = data["daily"]["temperature_2m_mean"][0]
                code = data["daily"]["weathercode"][0]
                
                # Bazi durumlarda API null dondurebilir
                if temp is None:
                    return None, "No Data (Null Temp)"
                    
                weather_desc = WEATHER_CODES.get(code, f"Unknown Code ({code})")
                return temp, weather_desc
            else:
                return None, "No Data (Empty API Response)"
        elif response.status_code == 429:
            return "RATE_LIMIT", "RATE_LIMIT"
        else:
            print(f"API Hatasi ({response.status_code}) - {response.text}")
            return None, f"API Error {response.status_code}"
    except Exception as e:
        print(f"Istek sirasinda hata olustu: {e}")
        return None, "Request Exception"

def process_earthquakes():
    # Dosya daha once islenmeye baslamissa oradan devam et, yoksa asil dosyayi ac
    if os.path.exists(OUTPUT_PATH):
        df = pd.read_csv(OUTPUT_PATH)
        print(f"Kaldigimiz yerden devam ediliyor. Mevcut kayit: {len(df)}")
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"Yeni baslaniyor. Toplam kayit: {len(df)}")
        # Yeni kolonlari olustur
        df["temperature"] = pd.NA
        df["weather_condition"] = pd.NA

    # Islem gerektiren satirlari bul
    # "No Data (Pre-1940)" gibi zaten tespit edilmis olanlari tekrar islememesi icin NaN kontrolu yapiyoruz
    to_process = df[df["weather_condition"].isna()]
    total_remaining = len(to_process)
    
    print(f"Islenmesi gereken toplam satir sayisi: {total_remaining}")
    
    # Her istekte calisacak dongu
    api_calls_made = 0
    BATCH_LIMIT = 5000  # Tek calistirmada API'ye kac kere soralim (Rate limit onlemi, 10 bin/gun limit var)
    
    try:
        for idx, row in to_process.iterrows():
            if api_calls_made >= BATCH_LIMIT:
                print(f"Guzel is! Bu turda {BATCH_LIMIT} istek yapildi, scripti dinlendirip veya yarın tekrar calistirabilirsin.")
                break

            lat = row["latitude"]
            lon = row["longitude"]
            time_str = row["time"]
            
            # API cagrisi
            temp, weather = get_weather(lat, lon, time_str)
            
            if temp == "RATE_LIMIT":
                print("Open-Meteo Rate Limit sinirina ulastik! Lutfen daha sonra tekrar deneyin (gunluk 10,000 siniri olabilir).")
                break
                
            # Sonuclari DataFrame'e kaydet
            df.at[idx, "temperature"] = temp
            df.at[idx, "weather_condition"] = weather
            api_calls_made += 1
            
            # Konsola kucuk bir cıktı
            if api_calls_made % 50 == 0:
                print(f"[{api_calls_made}/{total_remaining}] islendi... Sonuc: {time_str} | {lat},{lon} => Sicaklik: {temp}, Hava: {weather}")
                
            # Asiri yuklenmemek icin cok kisa bir bekleme
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nİşlem kullanıcı tarafından durduruldu. Veriler kaydediliyor...")
    finally:
        # Hata olsa bile mevcut veriyi kaydet (resume - kaldigin yerden devam edebilmek icin)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nİşlem bitti! Guncel veri '{OUTPUT_PATH}' konumuna kaydedildi.")
        print(f"Toplamda bu otorumda {api_calls_made} istek gerceklesti.")

if __name__ == "__main__":
    process_earthquakes()
