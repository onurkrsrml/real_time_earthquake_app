import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_gem = pd.read_csv("data/old_earthquakes.txt", sep="\t", comment="#")

dates = pd.DataFrame({
    'year': df_gem['Year'],
    'month': df_gem['Mo'].fillna(1).astype(int),
    'day': df_gem['Da'].fillna(1).astype(int),
    'hour': df_gem['Ho'].fillna(0).astype(int),
    'minute': df_gem['Mi'].fillna(0).astype(int),
    'second': df_gem['Se'].fillna(0).astype(int)
})

df_gem_filtered = pd.DataFrame()
df_gem_filtered['id'] = df_gem['En'].apply(lambda x: f"{x:.7f}")
df_gem_filtered['time'] = pd.to_datetime(dates, errors='coerce').astype('datetime64[ms]')
df_gem_filtered['magnitude'] = df_gem['M'].round(1)
df_gem_filtered['place'] = df_gem['Area']
df_gem_filtered['longitude'] = df_gem['Lon'].round(1)
df_gem_filtered['latitude'] = df_gem['Lat'].round(1)
df_gem_filtered['depth_km'] = df_gem['Dep'].round(1)
df_gem_filtered['source'] = 'GEM-GHEC'

df_usgs = pd.read_csv("data/earthquakes_1600_to_2026.csv")
df_usgs['time'] = pd.to_datetime(df_usgs['time'], errors='coerce').astype('datetime64[ms]')

combined_df = pd.concat([df_gem_filtered, df_usgs], ignore_index=True)
combined_df['time'] = combined_df['time'].astype('datetime64[ms]')
combined_df.to_csv("data/all_earthquakes_combined.csv", index=False)

print("Birleştirilmiş verisetinin ilk 5 satırı:")
print(combined_df.head())
print("\nBirleştirilmiş verisetinin son 5 satırı:")
print(combined_df.tail())
print("\nBirleştirilmiş verisetinin temel bilgileri:")
combined_df.info()
