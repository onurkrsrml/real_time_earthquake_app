# 🌍 DeepFault

## Yapay Zekâ Destekli Sismik Deprem Risk Skorlama Sistemi

### Spatio-Temporal Time Series Machine Learning Framework for Seismic Risk Estimation


# 📌 Proje Özeti

**DeepFault**, Türkiye ve çevresindeki aktif fay zonları üzerinde oluşabilecek kısa vadeli sismik risk yoğunluğunu analiz etmek amacıyla geliştirilmiş, 
ileri düzey **zaman serisi tabanlı yapay zekâ destekli deprem risk skorlama sistemidir.**

Bu proje klasik anlamda “deprem tahmini” üretmeyi hedeflemez. Bunun yerine;

* sismik aktivite yoğunluğu,
* enerji birikim davranışı,
* mekânsal kümelenme yapıları,
* zaman serisi anomalileri,
* çevresel ve jeofiziksel değişkenler

üzerinden **önümüzdeki 7 günlük süreçte belirli bölgelerde oluşabilecek sismik hareketlilik riskini olasılıksal olarak modellemektedir.**

Sistem; modern veri bilimi, istatistiksel sismoloji, zaman serisi mühendisliği ve gradient boosting algoritmalarını tek bir entegre mimaride birleştirmektedir.


# 🧠 Bilimsel Yaklaşım

Deprem oluşumu deterministik değil, yüksek derecede karmaşık ve çok değişkenli fiziksel süreçlerin sonucudur.

Bu nedenle DeepFault;

❌ “Şu gün deprem olacak” yaklaşımı yerine
✅ “Belirli bölgelerde kısa vadeli sismik risk yoğunluğu artıyor mu?” sorusuna odaklanır.

Model;

* geçmiş deprem dizilimleri,
* magnitüd dağılımları,
* enerji boşalımları,
* mikro-sismik yoğunluk artışları,
* zamansal ivmelenme davranışları

üzerinden istatistiksel risk örüntülerini öğrenmektedir.

# 🚀 Projenin Temel Hedefleri

## 🎯 Amaçlar

* Türkiye için AI destekli bölgesel sismik risk skoru üretmek
* Zaman serisi tabanlı deprem davranış örüntülerini modellemek
* Mikro sismik kümelenmeleri analiz etmek
* Bölgesel enerji birikim anomalilerini tespit etmek
* Gerçek zamanlı risk izleme altyapısı oluşturmak
* Bilimsel olarak leakage-free ML pipeline geliştirmek
* Açıklanabilir ve ölçeklenebilir bir deprem risk sistemi kurmak

# 🛰️ Kullanılan Veri Yapıları

Sistem yaklaşık:

## 📊 32.000+ Deprem Kaydı

üzerinde eğitilmiştir.

## 📅 Veri Aralığı

```text id="4m81r2"
1990 — 2026
```

# 🌐 Veri Kaynakları

## 1️⃣ Sismik Veriler

* Magnitüd
* Derinlik
* Latitude / Longitude
* Deprem sıklığı
* Enerji boşalımı
* Zamansal yoğunluk


## 2️⃣ Meteorolojik Veriler

* Atmosferik basınç
* Sıcaklık
* Bölgesel çevresel değişkenler

## 3️⃣ Astronomik Veriler

* Ay fazı
* Solar flux
* Güneş aktivitesi


## 4️⃣ Mekânsal Özellikler

* Grid-cell segmentation
* Fay hattı yakınlığı
* Bölgesel yoğunluk haritaları
* Spatial clustering


# ⚙️ Feature Engineering Mimarisi

DeepFault projesinin en güçlü bileşenlerinden biri ileri düzey feature engineering altyapısıdır.

Model içerisinde:

## 📈 27+ mühendislik özelliği

üretilmiştir.


## Kullanılan Başlıca Özellikler

### 🔹 Temporal Features

* Lag Features
* Rolling Mean
* Rolling Std
* Expanding Windows
* Time Decay Signals

### 🔹 Sismolojik Özellikler

* Gutenberg–Richter türevleri
* b-value analizi
* Enerji boşalım dinamikleri
* Magnitude acceleration
* Seismic momentum


### 🔹 Anomali Özellikleri

* Sismik volatilite
* Mikro-aktivite artışı
* Yoğunluk anomalileri
* Temporal pressure anomaly

### 🔹 Spatial Features

* Grid yoğunluğu
* Komşu hücre aktivitesi
* Bölgesel clustering skoru


# 🧪 Hedef Değişken (Target Definition)

Modelin hedef değişkeni şu şekilde tanımlanmıştır:

```python id="2kl8dp"
target = 1

Eğer aynı grid-cell içerisinde
önümüzdeki 7 gün içinde
M ≥ 4 büyüklüğünde deprem oluşursa
```

Aksi durumda:

```python id="c2s1z9"
target = 0
```

Bu yapı sayesinde model klasik binary classification problemi olarak optimize edilmiştir.


# 🛡️ Leakage-Free ML Pipeline

Zaman serisi projelerinde en kritik hata:

## ❌ Future Data Leakage

problemidir.

DeepFault mimarisi tamamen:

* geçmişe dayalı feature üretimi,
* kronolojik validasyon,
* walk-forward training

üzerine kurulmuştur.


## Kullanılan Güvenli Yapılar

```python id="lx0o0t"
shift(1).rolling(...)
```

Bu sayede modelin geleceğe ait bilgi sızdırması tamamen engellenmiştir.


# 🤖 Kullanılan Makine Öğrenmesi Algoritmaları

## Ana Model

### 🚀 XGBoost Classifier

Tercih edilme nedenleri:

* Non-linear pattern başarısı
* Temporal anomaly yakalama kapasitesi
* Yüksek feature interaction performansı
* Dengesiz veri başarımı
* Tabular data üstünlüğü


# 🎯 Hyperparameter Optimization

Model optimizasyonunda:

## ⚡ Optuna

kullanılmıştır.

Optimize edilen parametreler:

* learning_rate
* max_depth
* subsample
* colsample_bytree
* gamma
* min_child_weight
* regularization terms


# 📉 Validasyon Stratejisi

Klasik random split yaklaşımı kullanılmamıştır.

Çünkü zaman serilerinde random split:

❌ bilimsel olarak hatalıdır.


## Kullanılan Yaklaşım

### ✅ Walk-Forward Validation

```text id="tvb95y"
TRAIN  →  FUTURE TEST
```

Bu yaklaşım gerçek dünyadaki deprem akışını simüle eder.


# 📊 Model Çıktısı

Model aşağıdakileri üretir:

* Bölgesel risk olasılığı
* Risk skoru
* Sismik yoğunluk artışı
* Kısa vadeli anomaliler
* Potansiyel enerji birikim bölgeleri

# 🖥️ Streamlit Dashboard Sistemi

Proje içerisinde etkileşimli bir görselleştirme altyapısı bulunmaktadır.

## Özellikler

* Türkiye risk haritası
* İl bazlı skor analizi
* Bölgesel risk yoğunluğu
* İnteraktif filtreleme
* Gerçek zamanlı inference altyapısı


# 📂 Proje Yapısı

```text id="yk5waf"
DeepFault/
│
├── data/
├── notebooks/
├── models/
├── app.py
├── app_81_il.py
├── deepfault_Rabia_TIME_SERIES_V4_BEST_FINAL.py
├── requirements.txt
└── README.md
```


# 🧬 Ana Pipeline Akışı

```text id="q8rfz3"
Raw Earthquake Data
        ↓
Spatial Grid Construction
        ↓
Temporal Feature Engineering
        ↓
Leakage-Free Processing
        ↓
Walk-Forward Validation
        ↓
XGBoost Training
        ↓
Optuna Optimization
        ↓
Probability Calibration
        ↓
Risk Scoring Engine
        ↓
Streamlit Visualization
```

# 🔬 Bilimsel ve Teknik Güçlü Yönler

## ✅ Güçlü Taraflar

* Leakage-free architecture
* Spatio-temporal ML yaklaşımı
* İleri düzey feature engineering
* Gerçek dünya validasyonu
* Explainable risk production
* Modüler sistem mimarisi
* Ölçeklenebilir pipeline


# ⚠️ Kritik Bilimsel Not

Bu sistem:

## ❌ Deterministik deprem tahmin sistemi değildir.

DeepFault:

* istatistiksel risk yoğunluğu,
* sismik aktivite anomalisi,
* kısa vadeli bölgesel hareketlilik olasılığı

üretmektedir.

Bu nedenle çıktılar:

* erken uyarı sistemi,
* resmi afet tahmini,
* kesin deprem öngörüsü

olarak değerlendirilmemelidir.


# 🔮 Gelecek Çalışmalar

Planlanan geliştirmeler:

* LSTM / Transformer modelleri
* Graph Neural Networks
* Gerçek zamanlı AFAD & Kandilli entegrasyonu
* Uydu verisi entegrasyonu
* GPS deformasyon verileri
* Explainable AI modülü
* GPU distributed training
* Multi-country seismic learning


# 👩‍💻 Geliştirici

## Rabia AŞIK

### AI Researcher • Machine Learning Engineer • Time-Series Modeling • Seismic Risk Intelligence

Uzmanlık Alanları:

* Machine Learning
* Time Series Forecasting
* Earthquake Risk Modeling
* Spatio-Temporal AI Systems
* Feature Engineering
* Predictive Analytics


# 📜 Lisans

Bu proje:

* akademik araştırma,
* veri bilimi çalışmaları,
* deneysel sismik analiz

amaçlı geliştirilmiştir.

Sorumlu kullanım esastır.
