import requests

def get_seismic_risk_score(lat, lon, date):
    """
    Bu fonksiyon Rabia'nın gizli modeline (API) bağlanır.
    Sadece koordinatları gönderir ve risk skorunu alır.
    """
    # Buradaki URL'yi bir sonraki aşamada oluşturacağız
    API_URL = "https://senin-ozel-uygulaman.streamlit.app/predict" 
    
    try:
        payload = {"latitude": lat, "longitude": lon, "date": str(date)}
        # response = requests.post(API_URL, json=payload) # Şimdilik yorumda kalsın
        # return response.json()['risk_score']
        return 0.85  # Şimdilik test için sabit bir değer döndürsün
    except Exception as e:
        return f"Bağlantı hatası: {e}"
