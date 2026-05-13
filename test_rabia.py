def get_mock_rabia_data():
    print("⚠️ Rabia'nın API'si henüz HTML döndürdüğü için MOCK (Sahte) veri kullanılıyor...")
    
    # Rabia'nın gelecekte bize döndürmesini beklediğimiz tahmini veri formatı
    mock_veri = {
        "grid_id": "34_29",
        "region": "Marmara",
        "risk_score": 0.85,
        "risk_probability": 0.76
    }
    
    print("✅ BAŞARILI! Mock veri çekildi.")
    print("\n--- GELEN MOCK VERİ ---")
    for anahtar, deger in mock_veri.items():
        print(f"👉 {anahtar}: {deger}")
    print("-----------------------\n")
    
    return mock_veri

if __name__ == "__main__":
    get_mock_rabia_data()