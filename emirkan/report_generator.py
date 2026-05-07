def report_generator(combined_data):
    # Verileri alıyoruz
    region = combined_data.get("region")
    level = combined_data.get("alert_level", "UNKNOWN")
    score = combined_data.get("final_score", 0)
    consistency = combined_data["consistency_check"].get("consistency_score", 0)

    # 1. Teknik Rapor (Uzmanlar için)
    technical_report = (
        f"Model risk skoru ({score}), magnitüd beklentisi ve tutarlılık skoru ({consistency}) "
        f"birlikte değerlendirildiğinde bölgede {level} seviyesinde sismik aktivite olasılığı görülmektedir."
    )

    # 2. Halk İçin Sade Rapor
    public_report = (
        f"{region} bölgesinde olağan dışı sismik hareketlilik artışı gözlenmektedir. "
        f"Panik yapılmamalı, resmi kurumların açıklamaları takip edilmelidir."
    )

    # 3. Kurum / AFAD Dili
    institutional_report = (
        f"İlgili bölgede ({region}) izleme sıklığının artırılması, güncel sismik aktivitenin "
        f"takip edilmesi ve hazırlık seviyesinin gözden geçirilmesi önerilir."
    )

    return {
        "technical_report": technical_report,
        "public_report": public_report,
        "institutional_report": institutional_report,
        "raw_data": combined_data # Gerektiğinde n8n'e ham veriyi de basmak için
    }