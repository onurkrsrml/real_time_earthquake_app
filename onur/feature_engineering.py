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
import warnings
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

warnings.simplefilter(action="ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths & Constants
# ──────────────────────────────────────────────────────────────────────────────
INPUT_PATH = "data/depremler_hava_nasa.csv"
OUTPUT_PATH = "data/earthquakes_featured.csv"

RANDOM_SEED = 42
SEISMIC_WINDOW = "365D"
ENERGY_WINDOW = "180D"
VOLATILITY_WINDOW = "30D"
CLUSTER_WINDOW = "7D"
MC = 5.0


# ──────────────────────────────────────────────────────────────────────────────
# 0. Load Data
# ──────────────────────────────────────────────────────────────────────────────
def load_data(path=INPUT_PATH):
    """Load CSV, parse timestamps, and sort chronologically."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 1. Seismic Dynamics (Time-Varying Statistics)
# ──────────────────────────────────────────────────────────────────────────────
def _b_value(magnitudes):
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


def _gutenberg_deviation(magnitudes):
    """
    Difference between observed and expected earthquake count
    above the window-mean magnitude.
    """
    magnitudes = magnitudes[magnitudes >= MC]
    n = len(magnitudes)
    if n < 5:
        return np.nan
    mean_m = magnitudes.mean()
    mmin = magnitudes.min()
    denom = mean_m - mmin
    if denom <= 0:
        return np.nan
    b = np.log10(np.e) / denom
    expected_above_mean = n * 10 ** (-b * (mean_m - mmin))
    actual_above_mean = float(np.sum(magnitudes >= mean_m))
    return actual_above_mean - expected_above_mean


def add_seismic_dynamics(df):
    """Add 6 seismic dynamics features."""
    df = df.copy().set_index("time")
    log_energy = 1.5 * df["magnitude"] + 4.8

    df["b_value_trend"] = (
        df["magnitude"]
        .rolling(SEISMIC_WINDOW, min_periods=5)
        .apply(_b_value, raw=True)
    )

    df["depth_migration"] = (
        df["depth_km"]
        .rolling(SEISMIC_WINDOW, min_periods=5)
        .mean()
    )

    df["seismic_quietness_days"] = (
        df.index.to_series()
        .diff()
        .dt.total_seconds()
        .div(86400)
        .fillna(0)
    )

    df["energy_release_rolling"] = (
        log_energy
        .rolling(ENERGY_WINDOW, min_periods=1)
        .sum()
    )

    df["magnitude_volatility"] = (
        df["magnitude"]
        .rolling(VOLATILITY_WINDOW, min_periods=3)
        .std()
        .fillna(0)
    )

    df["gutenberg_deviation"] = (
        df["magnitude"]
        .rolling(SEISMIC_WINDOW, min_periods=5)
        .apply(_gutenberg_deviation, raw=True)
    )

    return df.reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# 2. Geophysical & Sensor-Based Data
# ──────────────────────────────────────────────────────────────────────────────
def add_geophysical_features(df):
    """Add 5 geophysical features (synthetic, physics-motivated)."""
    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(RANDOM_SEED)

    mag_std = df["magnitude"].std()
    mag_mean = df["magnitude"].mean()
    mag_norm = (df["magnitude"] - df["magnitude"].min()) / (
            df["magnitude"].max() - df["magnitude"].min() + 1e-9
    )
    depth_norm = (df["depth_km"] - df["depth_km"].min()) / (
            df["depth_km"].max() - df["depth_km"].min() + 1e-9
    )

    # Crustal Strain Rate
    base_strain = 50 + 10 * (df["magnitude"] - mag_mean) / (mag_std + 1e-9)
    df["crustal_strain_rate"] = (base_strain + rng.normal(0, 5, n)).clip(lower=0)

    # GPS Displacement Rate
    plate_speed = 20 + 5 * np.sin(np.radians(df["latitude"])) + 3 * np.cos(np.radians(df["longitude"]))
    df["gps_displacement_rate"] = (plate_speed + rng.normal(0, 2, n)).clip(lower=0)

    # Soil Radon Concentration
    df["soil_radon_concentration"] = (1000 + 500 * mag_norm + rng.normal(0, 80, n)).clip(lower=100)

    # Groundwater Level Change
    df["groundwater_level_change"] = -5 + 10 * (1 - depth_norm) + rng.normal(0, 3, n)

    # Electromagnetic Signal Power
    df["electromagnetic_signal_power"] = (
            30 + 0.05 * df["solar_flux_f107"] + 2 * (df["magnitude"] - 5) + rng.normal(0, 3, n)
    ).clip(lower=0)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Environmental & External Triggers
# ──────────────────────────────────────────────────────────────────────────────
def add_environmental_features(df):
    """Add 3 environmental trigger features."""
    df = df.copy()

    # Pressure Drop (Last 48h)
    df_time = df.set_index("time")
    rolling_min_48h = df_time["pressure"].rolling("2D", min_periods=1).min()
    df["pressure_drop_48h"] = (df_time["pressure"] - rolling_min_48h).values

    # Thermal Anomaly
    df["doy"] = df["time"].dt.dayofyear
    doy_mean = df.groupby("doy")["temperature"].transform("mean")
    df["thermal_anomaly"] = df["temperature"] - doy_mean
    df = df.drop(columns=["doy"])

    # Tidal Stress
    df["tidal_stress"] = np.cos(2 * np.pi * df["moon_phase"] / 100.0)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. Time Series Engineering
# ──────────────────────────────────────────────────────────────────────────────
def add_time_series_features(df):
    """Add 13 time series features (cyclic, lags, clustering)."""
    df = df.copy()

    # Cyclic Hour Encoding
    hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Cyclic Day-of-Year Encoding
    doy = df["time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Autoregressive Lags
    df_time = df.set_index("time")
    daily_mag = df_time["magnitude"].resample("1D").mean()
    daily_count = df_time["magnitude"].resample("1D").count()

    for lag_days in [1, 3, 7, 30]:
        shifted_mag = daily_mag.shift(lag_days)
        shifted_count = daily_count.shift(lag_days)

        mag_lag_col = f"mag_lag_{lag_days}d"
        count_lag_col = f"count_lag_{lag_days}d"

        lag_df = pd.DataFrame({"time": df["time"]})

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

        df[mag_lag_col] = merged_mag[mag_lag_col].values
        df[count_lag_col] = merged_cnt[count_lag_col].values

    # Temporal Clustering Index
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
def impute_missing_values(df):
    """Smart imputation for all engineered features."""
    df = df.copy()

    print("\n  [Imputation] Missing values BEFORE:")
    print(f"    b_value_trend: {df['b_value_trend'].isna().sum()}")
    print(f"    gutenberg_deviation: {df['gutenberg_deviation'].isna().sum()}")
    print(f"    depth_migration: {df['depth_migration'].isna().sum()}")
    for d in [1, 3, 7, 30]:
        print(f"    mag_lag_{d}d: {df[f'mag_lag_{d}d'].isna().sum()}")
        print(f"    count_lag_{d}d: {df[f'count_lag_{d}d'].isna().sum()}")

    # mag_lag_* features: Forward Fill → Global Mean → Backward Fill
    global_mag_mean = df["magnitude"].mean()
    for d in [1, 3, 7, 30]:
        col = f"mag_lag_{d}d"
        df[col] = df[col].ffill().fillna(global_mag_mean).bfill()

    # count_lag_* features: Forward Fill → Backward Fill
    for d in [1, 3, 7, 30]:
        col = f"count_lag_{d}d"
        df[col] = df[col].ffill().bfill()

    # depth_migration: Linear Interpolation
    df["depth_migration"] = df["depth_migration"].interpolate(
        method="linear", limit_direction="both"
    )

    # b_value_trend: Backward Fill → Interpolate → Global Mean
    df["b_value_trend"] = df["b_value_trend"].bfill()
    df["b_value_trend"] = df["b_value_trend"].interpolate(
        method="linear", limit_direction="both"
    )
    global_b_mean = df["b_value_trend"].mean()
    if np.isnan(global_b_mean):
        global_b_mean = 1.0
    df["b_value_trend"] = df["b_value_trend"].fillna(global_b_mean)

    # gutenberg_deviation: Backward Fill → Interpolate → 0
    df["gutenberg_deviation"] = df["gutenberg_deviation"].bfill()
    df["gutenberg_deviation"] = df["gutenberg_deviation"].interpolate(
        method="linear", limit_direction="both"
    )
    df["gutenberg_deviation"] = df["gutenberg_deviation"].fillna(0.0)

    print("\n  [Imputation] Missing values AFTER:")
    print(f"    b_value_trend: {df['b_value_trend'].isna().sum()}")
    print(f"    gutenberg_deviation: {df['gutenberg_deviation'].isna().sum()}")
    print(f"    depth_migration: {df['depth_migration'].isna().sum()}")
    for d in [1, 3, 7, 30]:
        print(f"    mag_lag_{d}d: {df[f'mag_lag_{d}d'].isna().sum()}")
        print(f"    count_lag_{d}d: {df[f'count_lag_{d}d'].isna().sum()}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6. Data Checking Function
# ──────────────────────────────────────────────────────────────────────────────
def check_df(dataframe, head=5):
    """Display basic DataFrame information."""
    print("\n##################### Shape #####################")
    print(dataframe.shape)
    print("\n##################### Types #####################")
    print(dataframe.dtypes)
    print("\n##################### Head #####################")
    print(dataframe.head(head))
    print("\n##################### Tail #####################")
    print(dataframe.tail(head))
    print("\n##################### NA #####################")
    print(dataframe.isnull().sum())


# ──────────────────────────────────────────────────────────────────────────────
# 7. Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_feature_engineering(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """
    Full feature-engineering pipeline.
    Reads raw earthquake CSV, adds 27 new features with smart imputation,
    and saves the enhanced dataset.
    """
    print("=" * 65)
    print("  FEATURE ENGINEERING PIPELINE")
    print("=" * 65)

    # Load Data
    print(f"\n[1/6] Loading data …  ({input_path})")
    df = load_data(input_path)
    print(f"      Rows: {len(df):,}   Columns: {df.shape[1]}")

    original_cols = set(df.columns)

    # Seismic Dynamics
    print("\n[2/6] Seismic Dynamics …")
    df = add_seismic_dynamics(df)

    # Geophysical & Sensor
    print("[3/6] Geophysical & Sensor features …")
    df = add_geophysical_features(df)

    # Environmental Triggers
    print("[4/6] Environmental triggers …")
    df = add_environmental_features(df)

    # Time Series Engineering
    print("[5/6] Time series engineering …")
    df = add_time_series_features(df)

    # Summary before imputation
    new_cols = [c for c in df.columns if c not in original_cols]
    print(f"\n{'─' * 65}")
    print(f"New features added: {len(new_cols)}")

    # Data Imputation
    print(f"\n{'─' * 65}")
    print("[6/6] Data imputation …")
    df = impute_missing_values(df)

    # Save
    df.to_csv(output_path, index=False)
    print(f"\n{'=' * 65}")
    print(f"✓ Enhanced dataset saved → {output_path}")
    print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'=' * 65}\n")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_enhanced = run_feature_engineering()
    print("✓ Feature engineering completed successfully!")