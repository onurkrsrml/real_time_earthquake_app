import pandas as pd
import ephem
from datetime import datetime
import warnings

warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

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

# 1. Veri setini oku
df = pd.read_csv("depremler_hava_nasa.csv")
df['time'] = pd.to_datetime(df['time'])

# 2. Ay Fazı (Illumination) Hesaplama
def get_moon_phase(date_obj, lat, lon):
    observer = ephem.Observer()
    observer.lat = str(lat)
    observer.lon = str(lon)
    observer.date = date_obj
    moon = ephem.Moon()
    moon.compute(observer)
    return moon.phase

df['moon_phase'] = df.apply(lambda row: get_moon_phase(row['time'], row['latitude'], row['longitude']), axis=1)
print("Ay fazı verileri başarıyla eklendi!")

# ---------------------------------------------------------
# GÜNEŞ AKTİVİTESİ (HİBRİT YAKLAŞIM) BÖLÜMÜ
# ---------------------------------------------------------
print("Güneş aktivitesi verileri indiriliyor...")

# 3. SILSO'dan Güneş Lekesi (Sunspot) verilerini çek
url_sunspot = "https://www.sidc.be/SILSO/DATA/SN_d_tot_V2.0.csv"
col_names = ['year', 'month', 'day', 'dec_year', 'sunspot_number', 'std_dev', 'obs_count', 'is_definitive']
solar_df = pd.read_csv(url_sunspot, sep=';', names=col_names)

# Tarih sütunu oluştur ve sadece gerekli sütunları tut
solar_df['date'] = pd.to_datetime(solar_df[['year', 'month', 'day']])
solar_df = solar_df[['date', 'sunspot_number']]

# Eksik sunspot verilerini temizle
solar_df.loc[solar_df['sunspot_number'] == -1, 'sunspot_number'] = None

# ADIM A: Formül ile tüm tarihler için F10.7 HESAPLA (Yedek Veri)
solar_df['f107_calc'] = 67.0 + (0.96 * solar_df['sunspot_number']) + (0.0031 * (solar_df['sunspot_number'] ** 2))

# ADIM B: CelesTrak API'den GERÇEK F10.7 ölçümlerini çek
print("Gerçek F10.7 verileri CelesTrak API'den alınıyor...")
url_f107 = "https://celestrak.org/SpaceData/SW-All.csv"
f107_api_df = pd.read_csv(url_f107)

# CelesTrak verilerini düzenle
f107_api_df['date'] = pd.to_datetime(f107_api_df['DATE'])
f107_api_df = f107_api_df[['date', 'F10.7_OBS']] # Gözlemlenen F10.7 verisi
f107_api_df.rename(columns={'F10.7_OBS': 'f107_real'}, inplace=True)
f107_api_df['f107_real'] = pd.to_numeric(f107_api_df['f107_real'], errors='coerce')

# ADIM C: İki güneş veri setini tarih üzerinden birleştir
solar_df = pd.merge(solar_df, f107_api_df, on='date', how='left')

# ADIM D: EKSİKLERİ TAMAMLA (Fillna Mantığı)
# Eğer 'f107_real' doluysa onu kullanır, NaN (boş) ise 'f107_calc' ile doldurur.
solar_df['solar_flux_f107'] = solar_df['f107_real'].fillna(solar_df['f107_calc'])

# Hesaplamada kullandığımız geçici sütunları kalabalık yapmaması için siliyoruz
solar_df.drop(columns=['f107_calc', 'f107_real'], inplace=True)

# ---------------------------------------------------------
# ANA VERİ SETİ İLE BİRLEŞTİRME
# ---------------------------------------------------------
print("Veriler birleştiriliyor...")

# Deprem veri setinden saati atıp sadece tarihi al
df['date_only'] = df['time'].dt.normalize()

# İki veri setini tarihlere göre birleştir
dff = pd.merge(df, solar_df, left_on='date_only', right_on='date', how='left')

# Geçici tarih sütunlarını temizle
dff.drop(columns=['date_only', 'date'], inplace=True)

print("İşlem tamamlandı! Eksiksiz hibrit 'solar_flux_f107' değişkeni oluşturuldu.")
# check_df(dff)

dff.to_csv("depremler_hava_nasa.csv", index=False)