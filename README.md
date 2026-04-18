# Real-Time Earthquake Data Analysis App

This project is a data processing pipeline that collects, cleans, and merges earthquake data from 1600 to the present from sources such as USGS (United States Geological Survey) and GEM-GHEC.

## 🚀 Features

- **Wide Time Range:** Covers earthquake data from 1600 to the present.
- **Smart Data Collection:** 
    - Historical data for pre-1900 with 5.5+ magnitude.
    - Modern data for post-1900 with 0.1+ magnitude.
- **Automatic Pagination and Chunking:** Safely downloads thousands of records without hitting USGS API limits.
- **Gap Filling:** Detects time gaps in the existing dataset and patches only the missing parts.
- **Data Merging:** Merges data from different sources (USGS and GEM) by bringing them into a standard format.

## 📁 Project Structure

```text
/real_time_earthquake_app
├── main.py                     <-- Main application entry point (Under development)
├── pyproject.toml              <-- Project settings and dependencies
├── uv.lock                     <-- Dependency lock file
├── README.md                   <-- Project documentation
├── LICENSE                     <-- License file
├── .env                        <-- Environment variables
│
├── data/                       <-- Data files
│   ├── earthquakes_1600_to_2026.csv    <-- Raw data fetched from USGS
│   ├── all_earthquakes_combined.csv     <-- Final merged dataset
│   └── old_earthquakes.txt             <-- GEM historical data source
│
├── onur/                       <-- Data Processing Modules
│   ├── api_data_collection.py   <-- USGS API client and data fetching logic
│   ├── old_earthquake_merge.py  <-- Script for merging datasets
│   └── main_onur.py             <-- Main script managing the entire data process
│
├── emirkan/                    <-- Emirkan's workspace
│   └── main_emirkan.py
│
└── rabia/                      <-- Rabia's workspace
    └── main_rabia.py
```

## 🛠 Installation

1. Clone the project:
   ```bash
   git clone https://github.com/user/real_time_earthquake_app.git
   cd real_time_earthquake_app
   ```

2. Install the required libraries:
   ```bash
   pip install requests pandas
   ```

## 💻 Usage

1. To start the entire data collection and merging process at once, simply run the `onur/main_onur.py` file:
```bash
python onur/main_onur.py
```

This command performs the following steps in order:
1. `api_data_collection.py` runs and updates data from USGS.
2. `old_earthquake_merge.py` runs and merges the data into the `all_earthquakes_combined.csv` file.

After the data processing is complete, you should run the following scripts in order:

2. Run the model training and comparison script:
   ```bash
   python rabia/main_rabia.py
   ```

3. Run the Streamlit application and AI agents:
   ```bash
   python emirkan/main_emirkan.py
   ```

## 👥 Contributors
- **Onur** - Data Scientist / Data Collection and Processing Pipeline
- **Rabia** - ML Engineer / Building and comparing models
- **Emirkan** - AI Engineer / Designing Streamlit and AI Agents
