import requests

def get_seismic_risk_score(lat, lon):
   
    API_URL = "https://deepfault-seismic-risk-scoring-jnzkmgvml4tlpmubcsuilb.streamlit.app" 
    
    try:
        # Bu kod senin sistemine "bu koordinatın skoru ne?" diye sorar
        response = requests.get(API_URL, params={"lat": lat, "lon": lon})
        
        # Senin sisteminden gelen cevabı alır
        result = response.json()
        return result["risk_score"]
    except Exception as e:
        # Eğer sistemine ulaşılamazsa bu hata döner
        return f"Sismik veri şu an çekilemiyor: {e}"
        
        # --- TEST ETMEK İÇİN (OPSİYONEL) ---
# Eğer bu dosyayı doğrudan çalıştırırsan bağlantıyı test eder:
#if __name__ == "__main__":
  #  print("Bağlantı test ediliyor...")
   # print(f"Test Skoru: {get_seismic_risk_score(37.5, 38.4)}")
