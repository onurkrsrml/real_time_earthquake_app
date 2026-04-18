import subprocess
import sys

def run_scripts():
    # 1. api_data_collection.py dosyasını çalıştır (USGS'den verileri çeker)
    print("\n>>> ADIM 1: api_data_collection.py çalıştırılıyor (USGS verileri toplanıyor)...")
    try:
        # Script'i python komutuyla çalıştırıyoruz
        subprocess.run([sys.executable, "onur/api_data_collection.py"], check=True)
        print(">>> ADIM 1 Tamamlandı.")
    except subprocess.CalledProcessError as e:
        print(f"!!! ADIM 1 HATA: api_data_collection.py çalışırken bir hata oluştu: {e}")
        return

    # 2. old_earthquake_merge.py dosyasını çalıştır (Verileri birleştirir)
    print("\n>>> ADIM 2: old_earthquake_merge.py çalıştırılıyor (Veriler birleştiriliyor)...")
    try:
        subprocess.run([sys.executable, "onur/old_earthquake_merge.py"], check=True)
        print(">>> ADIM 2 Tamamlandı.")
    except subprocess.CalledProcessError as e:
        print(f"!!! ADIM 2 HATA: old_earthquake_merge.py çalışırken bir hata oluştu: {e}")
        return

    print("\n=== TÜM İŞLEMLER BAŞARIYLA TAMAMLANDI ===")

if __name__ == "__main__":
    run_scripts()
