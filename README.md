# 🌍 AI-Powered Earthquake Risk & Early Warning System | Turkiye

Welcome to the **AI-Powered Earthquake Risk & Early Warning System** project! This study is designed to perform comprehensive earthquake analysis and develop a potential early warning system tailored for Turkiye. We aim for a multi-dimensional approach by integrating seismic data with atmospheric, meteorological, and astronomical variables.

---

## 📊 Dataset and Data Sources

Our dataset is a unique blend of seismic events, weather conditions during those events, and solar/lunar activities. The data preparation and processing steps were seamlessly executed via `onur/main_onur.py`, and the resulting dataset is stored in `data/depremler_hava_nasa.csv`. 

Currently, the data engineering phase is complete, and we have officially entered the analysis phase! 🚀

### 📡 Data Sources
- **🌋 USGS (United States Geological Survey):** Seismic data of earthquakes (location, magnitude, depth, etc.)
- **🌦️ NASA POWER API:** Weather and meteorological data at the exact time of the earthquake
- **🌒 Astronomical APIs & Calculations:** Moon phases and Solar activities (Sunspot counts, F10.7 flux)

---

## 🧬 Variables (Features)

Below is the data dictionary for our core dataset (`data/depremler_hava_nasa.csv`):

| Feature | Description |
| :--- | :--- |
| **`id`** | Unique record number of the earthquake in the system. |
| **`time`** | Exact date and time when the earthquake occurred. |
| **`magnitude`** | The magnitude / severity of the earthquake. |
| **`longitude`** | Longitude (East-West) coordinate of the epicenter. |
| **`latitude`** | Latitude (North-South) coordinate of the epicenter. |
| **`depth_km`** | Depth of the earthquake below the earth's surface in kilometers. |
| **`state`** | Province or administrative region where the earthquake is affiliated. |
| **`city_name`** | District or city center where the earthquake exactly occurred. |
| **`country`** | The country where the earthquake occurred. |
| **`moon_phase`** | The phase of the Moon on the date of the earthquake (illumination rate 0-100). |
| **`sunspot_number`** | The number of sunspots measured on that day (indicator of solar activity). |
| **`solar_flux_f107`**| The Solar radio flux (F10.7) measurement or calculated estimated value. |
| **`weather_desc`** | General weather condition (e.g., Clear, Partly Cloudy, Rainy). |
| **`temperature`** | Surface air temperature at the time of the earthquake (°C). |
| **`humidity`** | Relative humidity in the air at the time of the earthquake (%). |
| **`pressure`** | Atmospheric surface pressure at the time of the earthquake (hPa/millibars). |

---

## 🛠️ Next Steps (Project is Ongoing)

The data preparation phase is successfully completed! As this project is actively ongoing, our upcoming milestones include:
1. **🔍 Correlation Analysis:** Investigating the deep connections and potential relationships between seismic events and astronomical/meteorological data.
2. **🤖 Predictive Modeling:** Developing robust prediction and early warning models using advanced Machine Learning algorithms and Time Series Analysis.

---

## 👨‍💻 Developers

This project is brought to life by an enthusiastic team of data professionals:

- **Onur KARASÜRMELİ** - *Data Scientist* 
- **Rabia AŞIK** - *ML Engineer*
- **Emirkan EFE** - *AI Engineer*

---

*Stay tuned for more updates as we build a safer future with AI! 🛡️*
