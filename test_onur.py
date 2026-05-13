import os
import json

def test_onur_json():
    print("⏱️ Onur'un JSON çıktısı aranıyor...")
    
    # JSON dosyasının yolu (Onur'un koduna göre onur klasörünün içinde)
    dosya_yolu = os.path.join("onur", "prediction_output.json")
    
    if os.path.exists(dosya_yolu):
        try:
            with open(dosya_yolu, "r", encoding="utf-8") as f:
                veri = json.load(f)
            
            print("✅ BAŞARILI! Dosya bulundu ve okundu.")
            print("\n--- GELEN VERİ ---")
            for anahtar, deger in veri.items():
                print(f"👉 {anahtar}: {deger}")
            print("------------------\n")
            return True
            
        except Exception as e:
            print(f"❌ DOSYA BOZUK VEYA OKUNAMIYOR: {e}")
            return False
    else:
        print(f"❌ HATA: '{dosya_yolu}' bulunamadı!")
        print("💡 ÇÖZÜM: Lütfen önce terminalden 'python onur/main_onur.py' çalıştırıp dosyanın oluşmasını bekleyin.")
        return False

if __name__ == "__main__":
    test_onur_json()