def alert_classification_engine(rabia_output, onur_output):

    combined_data = {**rabia_output, **onur_output}

    risk = combined_data["risk_score"]
    buyukluk = combined_data["predicted_max_magnitude"]
    gun_sayisi = combined_data["days_to_event"]

    # threshold değerleri

    if risk >= 0.80 and buyukluk >= 5.5 and gun_sayisi <= 15:
        alarm_seviyesi = "Kritik (Kırmızı) Uyarı"
        aksiyon = "Acil Tahliye / Önlem Planı Başlatılmalı"
        
    elif risk >= 0.60 and buyukluk >= 4.5:
        alarm_seviyesi = "Yüksek (Turuncu) Uyarı"
        aksiyon = "Bölgesel Ekipler Hazırda Beklemeli"
        
    elif risk >= 0.30:
        alarm_seviyesi = "Orta (Sarı) Uyarı"
        aksiyon = "Rutin Gözlem ve Bilgilendirme"
        
    else:
        alarm_seviyesi = "Düşük (Yeşil) Uyarı"
        aksiyon = "Normal Durum"

    # 3. Aşama: Sonucu Yeni Bir Çıktı Olarak Hazırlama
    # Kendi motorumuzun ürettiği sonuçları birleştirilmiş veriye ekliyoruz
    combined_data["alert_level"] = alarm_seviyesi
    combined_data["recommended_action"] = aksiyon
    
    return combined_data

