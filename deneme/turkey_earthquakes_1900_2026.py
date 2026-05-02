import pandas as pd

from deneme import coordinate_finder as cf

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

def filter_turkey_earthquakes(df):
    if df is None or df.empty:
        return df

    filtered_df = df.copy()

    # Zaman (time) sütununu datetime'a çevirip 1900 yılı ve sonrasını filtrele
    if 'time' in filtered_df.columns:
        filtered_df['time'] = pd.to_datetime(filtered_df['time'], errors='coerce')
        filtered_df = filtered_df[filtered_df['time'] >= '1900-01-01']

    # Türkiye sınırlarına (koordinatlara) göre filtrele
    if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['latitude'] >= 34) & (filtered_df['latitude'] <= 44) &
            (filtered_df['longitude'] >= 24) & (filtered_df['longitude'] <= 47)
        ]
        
    return filtered_df

def load_dataframe():
    print("CSV dosyaları yükleniyor...")
    
    try:
        # df: data/all_earthquakes_combined.csv
        df = pd.read_csv("all_earthquakes_combined.csv")
        print(f"df (all_earthquakes_combined.csv) başarıyla yüklendi. Orijinal Boyut: {df.shape}")
    except FileNotFoundError:
        try:
            df = pd.read_csv("data/all_earthquakes_combined.csv")
            print(f"df (all_earthquakes_combined.csv) başarıyla yüklendi. Orijinal Boyut: {df.shape}")
        except FileNotFoundError:
            print("HATA: data/all_earthquakes_combined.csv dosyası bulunamadı!")
            df = None

    return df

df = load_dataframe()
df = filter_turkey_earthquakes(df)
df.dropna(inplace=True)

df_coordinated = cf.coordinate_finder(df, 'latitude', 'longitude')

if 'place' in df_coordinated.columns:
    # Virgülle ayırıp son elemanı 'country' olarak alma, strip ile boşlukları temizleme
    df_coordinated['country'] = df_coordinated['place'].apply(lambda x: x.split(',')[-1].strip() if ',' in x else x.strip())
    
    # "place" içindeki virgülün solunda kalan kısmı yeni "place" olarak bırakmak istiyoruz
    df_coordinated['place'] = df_coordinated['place'].apply(lambda x: x.split(',')[0].strip() if ',' in x else x.strip())

df_coordinated.drop(["source", "place"], axis=1, inplace=True)

df_coordinated['time'] = pd.to_datetime(df['time'])

if __name__ == "__main__":
        df_coordinated.to_csv("data/turkey_earthquakes_1900_2026.csv", index=False)
        print("İşlem Tamam! 'turkey_earthquakes_1900_2026.csv' dosyasına kaydedildi.")

