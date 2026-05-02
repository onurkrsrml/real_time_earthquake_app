########################################################################################################################
# __________________________________________ FEATURE ENGINEERING MODULE ______________________________________________
########################################################################################################################

"""
Feature Engineering Module – Real-Time Earthquake App
======================================================
Reads   : data/depremler_hava_nasa.csv
Outputs : data/earthquakes_featured.csv

New features are split into four categories:
  1. Seismic Dynamics          (6 features)
  2. Geophysical & Sensor Data (5 features)
  3. Environmental Triggers    (3 features)
  4. Time Series Engineering   (13 features: 4 cyclic + 8 lags + 1 cluster)
Total: 27 new columns.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Import Libraries
# ──────────────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action="ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths & Constants
# ──────────────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
INPUT_PATH  = os.path.join("data/depremler_hava_nasa.csv")
OUTPUT_PATH = os.path.join("data/earthquakes_featured.csv")

RANDOM_SEED      = 42
SEISMIC_WINDOW   = "365D"   # rolling window for seismic dynamics
ENERGY_WINDOW    = "180D"   # rolling window for energy release sum
VOLATILITY_WINDOW = "30D"   # rolling window for magnitude std-dev
CLUSTER_WINDOW   = "7D"     # rolling window for temporal clustering index

# Completeness magnitude (Mc) – lower bound used for b-value estimation
MC = 5.0


# ──────────────────────────────────────────────────────────────────────────────
# 0. Load Data
# ──────────────────────────────────────────────────────────────────────────────
def load_data(path: str = INPUT_PATH) -> pd.DataFrame:
    """Load CSV, parse timestamps, and sort chronologically."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df
load_data(path=INPUT_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Seismic Dynamics (Time-Varying Statistics)
# ──────────────────────────────────────────────────────────────────────────────
def _b_value(magnitudes: np.ndarray) -> float:
    """Aki (1965) maximum-likelihood b-value estimate."""
    magnitudes = magnitudes[magnitudes >= MC]
    n = len(magnitudes)
    if n < 5:
        return np.nan
    mean_m = magnitudes.mean()
    denom = mean_m - MC
    if denom <= 0:
        return np.nan
    return np.log10(np.e) / denom


def _gutenberg_deviation(magnitudes: np.ndarray) -> float:
    """
    Difference between the observed count of earthquakes above the window-mean
    magnitude and the count predicted by the Gutenberg-Richter relation.
    """
    magnitudes = magnitudes[magnitudes >= MC]
    n = len(magnitudes)
    if n < 5:
        return np.nan
    mean_m  = magnitudes.mean()
    mmin    = magnitudes.min()
    denom   = mean_m - mmin
    if denom <= 0:
        return np.nan
    b = np.log10(np.e) / denom
    expected_above_mean = n * 10 ** (-b * (mean_m - mmin))
    actual_above_mean   = float(np.sum(magnitudes >= mean_m))
    return actual_above_mean - expected_above_mean


def add_seismic_dynamics(df: pd.DataFrame) -> pd.DataFrame:
    """
    1.1  b_value_trend           – Gutenberg-Richter b-value (rolling)
    1.2  depth_migration         – Rolling mean focal depth (shallowing proxy)
    1.3  seismic_quietness_days  – Days since the previous earthquake
    1.4  energy_release_rolling  – Rolling log-energy sum (log10 Joules)
    1.5  magnitude_volatility    – 30-day rolling std-dev of magnitude
    1.6  gutenberg_deviation     – Observed minus expected count above mean M
    """
    df = df.copy().set_index("time")

    # Log-energy: E = 10^(1.5*M + 4.8)  →  log10(E) = 1.5*M + 4.8
    log_energy = 1.5 * df["magnitude"] + 4.8

    # 1.1 b-Value Trend
    df["b_value_trend"] = (
        df["magnitude"]
        .rolling(SEISMIC_WINDOW, min_periods=5)
        .apply(_b_value, raw=True)
    )

    # 1.2 Depth Migration
    df["depth_migration"] = (
        df["depth_km"]
        .rolling(SEISMIC_WINDOW, min_periods=5)
        .mean()
    )

    # 1.3 Seismic Quietness (days between consecutive events)
    df["seismic_quietness_days"] = (
        df.index.to_series()
        .diff()
        .dt.total_seconds()
        .div(86400)
        .fillna(0)
    )

    # 1.4 Energy Release Rolling Sum
    df["energy_release_rolling"] = (
        log_energy
        .rolling(ENERGY_WINDOW, min_periods=1)
        .sum()
    )

    # 1.5 Magnitude Volatility (Moving Std Dev)
    df["magnitude_volatility"] = (
        df["magnitude"]
        .rolling(VOLATILITY_WINDOW, min_periods=3)
        .std()
        .fillna(0)
    )

    # 1.6 Gutenberg Deviation
    df["gutenberg_deviation"] = (
        df["magnitude"]
        .rolling(SEISMIC_WINDOW, min_periods=5)
        .apply(_gutenberg_deviation, raw=True)
    )

    return df.reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Geophysical & Sensor-Based Data (Physics-Motivated Synthetic Values)
# ──────────────────────────────────────────────────────────────────────────────
def add_geophysical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    2.1  crustal_strain_rate         – Yer kabuğu deformasyon hızı (nanostrain/yr)
    2.2  gps_displacement_rate       – Tektonik plaka hızı (mm/yr)
    2.3  soil_radon_concentration    – Radon yoğunluğu (Bq/m³)
    2.4  groundwater_level_change    – Yeraltı su seviyesi değişimi (cm)
    2.5  electromagnetic_signal_power – VLF/ELF elektromanyetik güç (dB)

    Values are synthetic but physically motivated and reproducibly seeded.
    """
    df  = df.copy()
    n   = len(df)
    rng = np.random.default_rng(RANDOM_SEED)

    mag_std = df["magnitude"].std()
    mag_mean = df["magnitude"].mean()
    mag_norm = (df["magnitude"] - df["magnitude"].min()) / (
        df["magnitude"].max() - df["magnitude"].min() + 1e-9
    )
    depth_norm = (df["depth_km"] - df["depth_km"].min()) / (
        df["depth_km"].max() - df["depth_km"].min() + 1e-9
    )

    # 2.1 Crustal Strain Rate  – higher where magnitudes are elevated
    base_strain = 50 + 10 * (df["magnitude"] - mag_mean) / (mag_std + 1e-9)
    df["crustal_strain_rate"] = (
        base_strain + rng.normal(0, 5, n)
    ).clip(lower=0)

    # 2.2 GPS Displacement Rate  – driven by plate tectonic geometry
    plate_speed = (
        20
        + 5  * np.sin(np.radians(df["latitude"]))
        + 3  * np.cos(np.radians(df["longitude"]))
    )
    df["gps_displacement_rate"] = (
        plate_speed + rng.normal(0, 2, n)
    ).clip(lower=0)

    # 2.3 Soil Radon Concentration  – correlated with magnitude
    df["soil_radon_concentration"] = (
        1000 + 500 * mag_norm + rng.normal(0, 80, n)
    ).clip(lower=100)

    # 2.4 Groundwater Level Change  – shallow quakes cause larger changes
    df["groundwater_level_change"] = (
        -5 + 10 * (1 - depth_norm) + rng.normal(0, 3, n)
    )

    # 2.5 Electromagnetic Signal Power  – related to solar flux & magnitude
    df["electromagnetic_signal_power"] = (
        30
        + 0.05 * df["solar_flux_f107"]
        + 2    * (df["magnitude"] - 5)
        + rng.normal(0, 3, n)
    ).clip(lower=0)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Environmental & External Triggers
# ──────────────────────────────────────────────────────────────────────────────
def add_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    3.1  pressure_drop_48h – Atmosferik basınç düşüşü (son 48 saat, hPa)
    3.2  thermal_anomaly   – Yüzey sıcaklığının günün-yılı ortalamasından sapması (°C)
    3.3  tidal_stress      – Ay fazına dayalı gelgit stresi (normalize, -1..+1)

    Note: solar_flux_f107 is already present in the raw dataset.
    """
    df = df.copy()

    # ── 3.1 Pressure Drop over Last 48 h ──────────────────────────────────────
    df_time = df.set_index("time")
    rolling_min_48h = df_time["pressure"].rolling("2D", min_periods=1).min()
    df["pressure_drop_48h"] = (df_time["pressure"] - rolling_min_48h).values

    # ── 3.2 Thermal Anomaly ───────────────────────────────────────────────────
    df["doy"] = df["time"].dt.dayofyear
    doy_mean  = df.groupby("doy")["temperature"].transform("mean")
    df["thermal_anomaly"] = df["temperature"] - doy_mean
    df = df.drop(columns=["doy"])

    # ── 3.3 Tidal Stress ──────────────────────────────────────────────────────
    # moon_phase in the dataset is illumination percentage (0–100).
    # Full moon (100) and new moon (0) both produce peak tidal stress;
    # quarter moons (≈50) produce minimum stress.
    # cos(2π · phase/100) maps 0→+1, 50→−1, 100→+1  (two cycles per month).
    df["tidal_stress"] = np.cos(2 * np.pi * df["moon_phase"] / 100.0)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. Time Series Engineering
# ──────────────────────────────────────────────────────────────────────────────
def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    4.1  hour_sin / hour_cos   – Dairesel saat kodlaması
    4.2  doy_sin  / doy_cos    – Dairesel yılın-günü kodlaması
    4.3  mag_lag_{n}d          – t-1, t-3, t-7, t-30 gün geçmiş ortalama magnitüd
    4.4  count_lag_{n}d        – t-1, t-3, t-7, t-30 gün geçmiş deprem sayısı
    4.5  temporal_clustering_index – Son 7 günlük deprem yoğunluğu
    """
    df = df.copy()

    # ── 4.1 Cyclic Hour Encoding ──────────────────────────────────────────────
    hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # ── 4.2 Cyclic Day-of-Year Encoding ──────────────────────────────────────
    doy = df["time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # ── 4.3 & 4.4  Autoregressive Lags ───────────────────────────────────────
    df_time = df.set_index("time")

    # Build daily aggregate series (mean magnitude, count per day)
    daily_mag   = df_time["magnitude"].resample("1D").mean()
    daily_count = df_time["magnitude"].resample("1D").count()

    for lag_days in [1, 3, 7, 30]:
        shifted_mag   = daily_mag.shift(lag_days)
        shifted_count = daily_count.shift(lag_days)

        # Align back to original timestamps via forward-fill (asof merge)
        mag_lag_col   = f"mag_lag_{lag_days}d"
        count_lag_col = f"count_lag_{lag_days}d"

        # Use merge_asof for efficient alignment
        lag_df = pd.DataFrame({
            "time": df["time"],
        })
        lookup_mag = shifted_mag.reset_index()
        lookup_mag.columns = ["date", mag_lag_col]
        lookup_mag["date"] = pd.to_datetime(lookup_mag["date"])

        lookup_cnt = shifted_count.reset_index()
        lookup_cnt.columns = ["date", count_lag_col]
        lookup_cnt["date"] = pd.to_datetime(lookup_cnt["date"])

        merged_mag = pd.merge_asof(
            lag_df.sort_values("time"),
            lookup_mag,
            left_on="time",
            right_on="date",
            direction="backward",
        ).sort_index()

        merged_cnt = pd.merge_asof(
            lag_df.sort_values("time"),
            lookup_cnt,
            left_on="time",
            right_on="date",
            direction="backward",
        ).sort_index()

        df[mag_lag_col]   = merged_mag[mag_lag_col].values
        df[count_lag_col] = merged_cnt[count_lag_col].values

    # ── 4.5 Temporal Clustering Index ────────────────────────────────────────
    df["temporal_clustering_index"] = (
        df_time["magnitude"]
        .rolling(CLUSTER_WINDOW, min_periods=1)
        .count()
        .values
    )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5. Data Imputation
# ──────────────────────────────────────────────────────────────────────────────
def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smart imputation for the 27 engineered features.

    Strategies
    ----------
    mag_lag_{1,3,7,30}d
        Forward-fill (last valid magnitude average repeated) →
        fallback to global magnitude mean →
        final fallback: backward-fill.

    count_lag_{1,3,7,30}d
        Forward-fill then backward-fill.
        (minimal missingness, trend is preserved)

    depth_migration
        Linear interpolation with limit_direction='both'.
        (continuous physical variable, smooth transition)

    b_value_trend
        Step 1 – backward-fill (slow-changing stress indicator, old value repeatable)
        Step 2 – linear interpolation (smooth transition)
        Step 3 – global b-value mean (final fallback)

    gutenberg_deviation
        Step 1 – backward-fill (trend continues)
        Step 2 – linear interpolation
        Step 3 – 0.0 (null hypothesis: observed == expected)
    """
    df = df.copy()

    IMPUTED_COLS = (
        [f"mag_lag_{d}d"   for d in [1, 3, 7, 30]]
        + [f"count_lag_{d}d" for d in [1, 3, 7, 30]]
        + ["depth_migration", "b_value_trend", "gutenberg_deviation"]
    )

    # ── Before report ─────────────────────────────────────────────────────────
    print("\n  [Imputation] Missing values BEFORE:")
    for col in IMPUTED_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            print(f"    {col:<35}  nulls={n:,}")

    # ── 1. mag_lag_{1,3,7,30}d  – Forward Fill → global mean → backward fill ──
    global_mag_mean = df["magnitude"].mean()
    for d in [1, 3, 7, 30]:
        col = f"mag_lag_{d}d"
        if col not in df.columns:
            continue
        df[col] = df[col].ffill().fillna(global_mag_mean).bfill()

    # ── 2. count_lag_{1,3,7,30}d  – Forward Fill → backward fill ─────────────
    for d in [1, 3, 7, 30]:
        col = f"count_lag_{d}d"
        if col not in df.columns:
            continue
        df[col] = df[col].ffill().bfill()

    # ── 3. depth_migration  – Linear Interpolation ────────────────────────────
    if "depth_migration" in df.columns:
        df["depth_migration"] = df["depth_migration"].interpolate(
            method="linear", limit_direction="both"
        )

    # ── 4. b_value_trend  – bfill → interpolate → global b-value mean ─────────
    if "b_value_trend" in df.columns:
        df["b_value_trend"] = (
            df["b_value_trend"]
            .bfill()
            .interpolate(method="linear", limit_direction="both")
        )
        global_b_mean = df["b_value_trend"].mean()
        if np.isnan(global_b_mean):
            global_b_mean = 1.0   # typical seismic b-value
        df["b_value_trend"] = df["b_value_trend"].fillna(global_b_mean)

    # ── 5. gutenberg_deviation  – bfill → interpolate → 0 ────────────────────
    if "gutenberg_deviation" in df.columns:
        df["gutenberg_deviation"] = (
            df["gutenberg_deviation"]
            .bfill()
            .interpolate(method="linear", limit_direction="both")
            .fillna(0.0)
        )

    # ── After report ──────────────────────────────────────────────────────────
    print("\n  [Imputation] Missing values AFTER:")
    for col in IMPUTED_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            print(f"    {col:<35}  nulls={n:,}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6. Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_feature_engineering(
    input_path:  str = INPUT_PATH,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Full feature-engineering pipeline.
    Reads the raw earthquake CSV, adds 27 new columns, and writes the
    enhanced dataset to *output_path*.

    Returns the enhanced DataFrame.
    """
    print("=" * 65)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 65)

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print(f"\n[1/6] Loading data …  ({input_path})")
    df = load_data(input_path)
    print(f"      Rows: {len(df):,}   Columns: {df.shape[1]}")

    original_cols = set(df.columns)

    # ── Step 2: Seismic Dynamics ──────────────────────────────────────────────
    print("\n[2/6] Seismic Dynamics (b-value, depth migration, quietness, "
          "energy, volatility, GR deviation) …")
    df = add_seismic_dynamics(df)

    # ── Step 3: Geophysical & Sensor ──────────────────────────────────────────
    print("[3/6] Geophysical & Sensor features (strain, GPS, radon, "
          "groundwater, EM) …")
    df = add_geophysical_features(df)

    # ── Step 4: Environmental Triggers ────────────────────────────────────────
    print("[4/6] Environmental triggers (pressure drop, thermal anomaly, "
          "tidal stress) …")
    df = add_environmental_features(df)

    # ── Step 5: Time Series Engineering ───────────────────────────────────────
    print("[5/6] Time series engineering (cyclic encoding, lags, clustering) …")
    df = add_time_series_features(df)

    # ── Summary (before imputation) ───────────────────────────────────────────
    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"\n{'─' * 65}")
    print(f"New features added: {len(new_cols)}")
    for col in new_cols:
        nulls = df[col].isna().sum()
        print(f"  {col:<35}  nulls={nulls:,}")

    # ── Step 6: Data Imputation ───────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("[6/6] Data imputation (smart fill for all 27 new features) …")
    df = impute_missing_values(df)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n{'=' * 65}")
    print(f"  Enhanced dataset saved → {output_path}")
    print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'=' * 65}\n")

    return df

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    # print("##################### Quantiles #####################")
    # print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_enhanced = run_feature_engineering()
