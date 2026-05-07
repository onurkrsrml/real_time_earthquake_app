# test_cases.py

# SENARYO 1: Kritik Durum (Daha önce yaptığımız)
rabia_test_1 = {
    "grid_id": "34_29", 
    "region": "Marmara", 
    "risk_score": 0.85, 
    "risk_probability": 0.76
}
onur_test_1 = {
    "days_to_event": 5, 
    "predicted_max_magnitude": 5.8, 
    "confidence_score": 0.78
}

# SENARYO 2: Düşük Risk (Sakin Tablo)
# Burada hem Rabia'nın riski düşük hem de Onur'un magnitüdü ve aciliyeti az.
rabia_test_2 = {
    "grid_id": "06_01", 
    "region": "İç Anadolu", 
    "risk_score": 0.15, 
    "risk_probability": 0.20
}
onur_test_2 = {
    "days_to_event": 60,               # Olay çok uzak
    "predicted_max_magnitude": 2.1,    # Çok küçük sarsıntı
    "confidence_score": 0.40           # Güven düşük
}

# SENARYO 3: Model Çelişkisi (Tutarlılık Motorunu Test Etmek İçin)
# Rabia risk yok diyor ama Onur büyük deprem geliyor diyor!
rabia_test_3 = {
    "grid_id": "35_09", 
    "region": "Ege", 
    "risk_score": 0.20, 
    "risk_probability": 0.15
}
onur_test_3 = {
    "days_to_event": 3, 
    "predicted_max_magnitude": 6.8, 
    "confidence_score": 0.85
}