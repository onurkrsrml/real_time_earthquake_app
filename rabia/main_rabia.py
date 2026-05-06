ef get_seismic_risk_score(latitude, longitude, time):
    """
    RABİA - Sismik Risk Modeli Girdi Formatı
    Girdiler: 
    - latitude: float (Örn: 37.5)
    - longitude: float (Örn: 38.4)
    - time: string (Örn: "2026-05-06")
    """
   
    API_URL = "https://deepfault-seismic-risk-scoring-jnzkmgvml4tlpmubcsuilb.streamlit.app" 
    
    params = {
        "latitude": latitude, 
        "longitude": longitude, 
        "time": time
    }
    
    try:
        response = requests.get(API_URL, params=params)
        return response.json()["risk_score"]
    except:
        return "Bağlantı hatası: Veri çekilemedi."
        
        # --- TEST ETMEK İÇİN (OPSİYONEL) ---
# Eğer bu dosyayı doğrudan çalıştırırsan bağlantıyı test eder:
#if __name__ == "__main__":
  #  print("Bağlantı test ediliyor...")
   # print(f"Test Skoru: {get_seismic_risk_score(37.5, 38.4)}")
