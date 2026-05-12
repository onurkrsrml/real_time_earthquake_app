########################################################################################################################
# ============================================ EARLY WARNING EARTHQUAKE APP ============================================
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
# COLUMNS
# ----------------------------------------------------------------------------------------------------------------------
#
# id: Unique record number of the earthquake in the system.
# time: Exact date and time when the earthquake occurred.
# magnitude: The magnitude / severity of the earthquake.
# longitude: Longitude (East-West) coordinate of the epicenter.
# latitude:	Latitude (North-South) coordinate of the epicenter.
# depth_km:	Depth of the earthquake below the earth's surface in kilometers.
# state: Province or administrative region where the earthquake is affiliated.
# city_name: District or city center where the earthquake exactly occurred.
# country: The country where the earthquake occurred.
# moon_phase: The phase of the Moon on the date of the earthquake (illumination rate 0-100).
# sunspot_number: The number of sunspots measured on that day (indicator of solar activity).
# solar_flux_f107: The Solar radio flux (F10.7) measurement or calculated estimated value.
# weather_desc: General weather condition (e.g., Clear, Partly Cloudy, Rainy).
# temperature: Surface air temperature at the time of the earthquake (°C).
# humidity: Relative humidity in the air at the time of the earthquake (%).
# pressure: Atmospheric surface pressure at the time of the earthquake (hPa/millibars).
# region: Short names of the fault lines where the earthquake occurred (KAF, DAF, BAF)


# ----------------------------------------------------------------------------------------------------------------------
# CONTENTS
# ----------------------------------------------------------------------------------------------------------------------

# 1. EDA
#   Import Libraries
#   Display Options
#   Warnings Ignore
#   Load Df
#   Required Changes
#   Column Rechanging
#   Check Df
#   Safe Rolling
#
# 2. FEATURE ENGINEERING
#   Grab Col Names
#   Outlier Thresholds
#   Check Outlier
#   Replace With Thresholds
#   Sorting to prevent Data leakage
#   Interpolate & Bfill
#
# 3. FEATURE SCALING
#   Standard Scaler
#
# 4. FEATURE EXTRACTION
#   Time Features
#   Target 1: future_max_magnitude
#   Target 2: days_to_next_major_event
#   Dropna in Targets
#
# 5. FEATURE INTERACTIONS
#   New Features
#
# 6. MODELLING
#   Model Preparation
#   Walk-Forward Validation (TimeSeriesSplit)
#   Initialize Models
#   Model Evaluations
#
# 7. FEATURE IMPORTANCE
#   Plot Importance Function
#   Train Models
#   Plot Importance
#
# 8. HYPERPARAMETER OPTIMIZATION
#   Hyperparameter Optimization
#
# 9. FINAL MODEL
#   Generate Prediction JSON
#
# 10. EXPLAINABILITY AND PRODUCT DEVELOPMENT
#   Generate Prediction JSON
#
# 11. MAIN FUNCTION (EXECUTION)
#   Main Function
#   Run Function
#


print("""\n
      ___________________________________________ EARLY WARNING EARTHQUAKE APP _________________________________________

        --------
        COLUMNS
        --------
        id:	Unique record number of the earthquake in the system.
        time	: Exact date and time when the earthquake occurred.
        magnitude:	The magnitude / severity of the earthquake.
        longitude:	Longitude (East-West) coordinate of the epicenter.
        latitude:	Latitude (North-South) coordinate of the epicenter.
        depth_km:	Depth of the earthquake below the earth's surface in kilometers.
        state:	Province or administrative region where the earthquake is affiliated.
        city_name:	District or city center where the earthquake exactly occurred.
        country:	The country where the earthquake occurred.
        moon_phase:	The phase of the Moon on the date of the earthquake (illumination rate 0-100).
        sunspot_number:	The number of sunspots measured on that day (indicator of solar activity).
        solar_flux_f107:	The Solar radio flux (F10.7) measurement or calculated estimated value.
        weather_desc:	General weather condition (e.g., Clear, Partly Cloudy, Rainy).
        temperature:	Surface air temperature at the time of the earthquake (°C).
        humidity:	Relative humidity in the air at the time of the earthquake (%).
        pressure:	Atmospheric surface pressure at the time of the earthquake (hPa/millibars).
        region: Short names of the fault lines where the earthquake occurred (KAF, DAF, BAF)


        --------
        CONTENTS
        --------
        1. EDA
            Import Libraries
            Display Options
            Warnings Ignore
            Load Df
            Required Changes
            Column Rechanging
            Check Df
            Safe Rolling

        2. FEATURE ENGINEERING
            Grab Col Names
            Outlier Thresholds
            Check Outlier
            Replace With Thresholds
            Sorting to prevent Data leakage
            Interpolate & Bfill

        3. FEATURE SCALING
            Standard Scaler

        4. FEATURE EXTRACTION
            Time Features
            Target 1: future_max_magnitude
            Target 2: days_to_next_major_event
            Dropna in Targets

        5. FEATURE INTERACTIONS
            New Features

        6. MODELLING
            Model Preparation
            Walk-Forward Validation (TimeSeriesSplit)
            Initialize Models
            Model Evaluations

        7. FEATURE IMPORTANCE
            Plot Importance Function
            Train Models
            Plot Importance

        8. HYPERPARAMETER OPTIMIZATION
            Hyperparameter Optimization

        9. FINAL MODEL
            Generate Prediction JSON

        10. EXPLAINABILITY AND PRODUCT DEVELOPMENT
            Generate Prediction JSON

        11. MAIN FUNCTION (EXECUTION)
            Main Function
            Run Function
            
      __________________________________________________________________________________________________________________
      \n""")


########################################################################################################################
# STEP 1 : EDA
########################################################################################################################
print("\n", "STEP 1 : EDA", "\n", "=" * 30, "\n")


# Import Libraries
print("\n", "Import Libraries", "\n", "_" * 30, "\n")
import pandas as pd
import numpy as np
import os
import json
import re
import joblib
from datetime import date
import optuna
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings


# Display Options
print("\n", "Display Options", "\n", "_" * 30, "\n")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


# Warnings Ignore
print("\n", "Warnings Ignore", "\n", "_" * 30, "\n")
warnings.simplefilter(action="ignore")


# Load Df
print("\n", "Load Df", "\n", "_" * 30, "\n")
def load_df():
    data = pd.read_csv("data/earthquakes_featured.csv", low_memory=False)
    return data
df = load_df()


# Required Changes
print("\n", "Required Changes", "\n", "_" * 30, "\n")
df['time'] = pd.to_datetime(df['time'])
if "id" in df.columns:
    df = df.drop(["id"], axis=1)
if "Unnamed: 0" in df.columns:
    df = df.drop(["Unnamed: 0"], axis=1)


# Column Rechanging
required_cols = {
    "date", "latitude", "longitude", "depth", "magnitude", "grid_id", "region",
    "energy", "event_count"}
optional_cols = {
    "pressure", "temperature", "moon_phase", "solar_flux", "fault_distance",
    "rolling_energy", "rolling_magnitude", "b_value"}
column_aliases = {
    "time": "date",
    "depth_km": "depth",
    "solar_flux_f107": "solar_flux",
    "b_value_trend": "b_value"}

df.rename(columns={k: v for k, v in column_aliases.items() if k in df.columns}, inplace=True)
missing_required = required_cols - set(df.columns)
missing_optional = optional_cols - set(df.columns)


# Check Df
print("\n", "Check Df", "\n", "_" * 30, "\n")
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
check_df(df)


# Safe Rolling
def safe_rolling(group, col, window, func="mean"):
    shifted = group[col].shift(1)
    if func == "mean":
        return shifted.rolling(window, min_periods=1).mean()
    if func == "max":
        return shifted.rolling(window, min_periods=1).max()
    if func == "std":
        return shifted.rolling(window, min_periods=1).std()
    if func == "sum":
        return shifted.rolling(window, min_periods=1).sum()
    return shifted.rolling(window, min_periods=1).mean()


########################################################################################################################
# STEP 2 : FEATURE ENGINEERING
########################################################################################################################
print("\n", "STEP 2 : FEATURE ENGINEERING", "\n", "=" * 30, "\n")


# Grab Col Names
print("\n", "Grab Col Names", "\n", "_" * 30, "\n")
def grab_col_names(dataframe, cat_th=10, car_th=20, show_names=False):
    cat_cols = dataframe.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_but_cat = [col for col in dataframe.columns if
                   (dataframe[col].nunique() < cat_th and dataframe[col].dtype in ["int64", "float64"])]
    cat_but_car = [col for col in dataframe.columns if
                   (dataframe[col].nunique() > car_th and dataframe[col].dtype in ["object", "string", "category"])]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = dataframe.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols = [col for col in num_cols if col not in num_but_cat]
    id_cols = [col for col in dataframe.columns if re.search(r"\bid\b", col, re.IGNORECASE)]
    date_cols = [col for col in dataframe.columns if pd.api.types.is_datetime64_any_dtype(dataframe[col])]
    num_cols = [col for col in num_cols if (col not in id_cols and col not in date_cols)]
    return cat_cols, num_cols, cat_but_car, num_but_cat, id_cols, date_cols
cat_cols, num_cols, cat_but_car, num_but_cat, id_cols, date_cols = grab_col_names(df)


# Outlier Thresholds
print("\n", "Outlier Thresholds", "\n", "_" * 30, "\n")
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Check Outlier
print("\n", "Check Outlier", "\n", "_" * 30, "\n")
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].shape[0] > 0:
        return True
    return False


# Replace With Thresholds
print("\n", "Replace With Thresholds", "\n", "_" * 30, "\n")
for col in num_cols:
    if check_outlier(df, col):
        low, up = outlier_thresholds(df, col)
        df.loc[(df[col] < low), col] = low
        df.loc[(df[col] > up), col] = up

# Sorting to prevent Data leakage
print("\n", "Sorting to prevent Data leakage", "\n", "_" * 30, "\n")
df = df.sort_values(by=['date']).reset_index(drop=True)


# Interpolate & Bfill
print("\n", "Bfill", "\n", "_" * 30, "\n")
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df[num_cols] = df[num_cols].interpolate(method='time').bfill()
df.reset_index(inplace=True)


########################################################################################################################
# STEP 3 : FEATURE SCALING
########################################################################################################################
print("\n", "STEP 3 : FEATURE SCALING", "\n", "=" * 30, "\n")


# Standard Scaler
print("\n", "Standard Scaler", "\n", "_" * 30, "\n")
scaler = StandardScaler()
for col in num_cols:
    if col not in ['magnitude', 'depth']:
        df[col + '_scaled'] = scaler.fit_transform(df[[col]])


########################################################################################################################
# STEP 4 : FEATURE EXTRACTION & TARGET DEFINITION
########################################################################################################################
print("\n", "STEP 4 : FEATURE EXTRACTION & TARGET DEFINITION", "\n", "=" * 30, "\n")


# Time Features
print("\n", "Time Features", "\n", "_" * 30, "\n")
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek

if "grid_id" not in df.columns:
    if "region" in df.columns:
        df["grid_id"] = df["region"].astype(str)
    else:
        df["grid_id"] = "grid_00"

MAJOR_MAG = 4.0


# Target 1: future_max_magnitude_7d
print("\n", "Target 1: future_max_magnitude_7d", "\n", "_" * 30, "\n")
def compute_future_max_mag_7d(group):
    times = group["date"].values
    mags = group["magnitude"].values
    result = np.zeros(len(group))

    for i in range(len(group)):
        future_mask = (times > times[i]) & (times <= times[i] + np.timedelta64(7, "D"))
        if future_mask.any():
            result[i] = mags[future_mask].max()
        else:
            result[i] = 0.0
    return pd.Series(result, index=group.index)
df["future_max_magnitude_7d"] = df.groupby("grid_id", group_keys=False).apply(compute_future_max_mag_7d).reset_index(level=0, drop=True)
df["future_max_magnitude_7d"] = df["future_max_magnitude_7d"].values


# Target 2: days_to_next_major_event
print("\n", "Target 2: days_to_next_major_event", "\n", "_" * 30, "\n")
def compute_days_to_next_major(group):
    times = group["date"].values
    mags = group["magnitude"].values
    result = np.full(len(group), np.nan)

    for i in range(len(group)):
        future_mask = mags[i+1:] >= MAJOR_MAG
        if future_mask.any():
            next_idx = i + 1 + np.argmax(future_mask)
            delta_days = (times[next_idx] - times[i]) / np.timedelta64(1, "D")
            result[i] = delta_days
        else:
            result[i] = 60
    return pd.Series(result, index=group.index)
df["days_to_next_major_event"] = df.groupby("grid_id", group_keys=False).apply(compute_days_to_next_major).reset_index(level=0, drop=True)
df["days_to_next_major_event"] = df["days_to_next_major_event"].values


# Dropna in Targets
print("\n", "Dropna in Targets", "\n", "_" * 30, "\n")
df.dropna(subset=['future_max_magnitude_7d', 'days_to_next_major_event'], inplace=True)
df.reset_index(drop=True, inplace=True)


########################################################################################################################
# STEP 5 : FEATURE INTERACTIONS
########################################################################################################################
print("\n", "STEP 5 : FEATURE INTERACTIONS", "\n", "=" * 30, "\n")


# New Features
print("\n", "New Features", "\n", "_" * 30, "\n")
df["season"] = df["month"] % 12 // 3 + 1

df["days_since_last_event"] = df.groupby("grid_id")["date"].diff().dt.days
df["days_since_last_major_event"] = df.groupby("grid_id").apply(
    lambda g: g["date"].where(g["magnitude"] >= 4.0).ffill()).reset_index(level=0, drop=True)
df["days_since_last_major_event"] = (df["date"] - df["days_since_last_major_event"]).dt.days

if "event_count" not in df.columns:
    df["event_count"] = 1
for w in [7, 14, 30]:
    df[f"event_frequency_last_{w}d"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: safe_rolling(g, "event_count", w, func="sum"))

for w in [7, 14, 30]:
    df[f"rolling_mean_magnitude_{w}d"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: safe_rolling(g, "magnitude", w, func="mean"))
    df[f"rolling_max_magnitude_{w}d"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: safe_rolling(g, "magnitude", w, func="max"))
df["magnitude_std_7d"] = df.groupby("grid_id", group_keys=False).apply(
    lambda g: safe_rolling(g, "magnitude", 7, func="std"))
df["magnitude_acceleration"] = df["rolling_mean_magnitude_7d"] - df["rolling_mean_magnitude_14d"]
df["magnitude_trend"] = df["rolling_mean_magnitude_7d"] - df["rolling_mean_magnitude_30d"]

df["energy"] = 10 ** (1.5 * df["magnitude"] + 4.8)
for w in [7, 14, 30]:
    df[f"rolling_energy_sum_{w}d"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: safe_rolling(g, "energy", w, func="sum"))
df["energy_acceleration"] = df["rolling_energy_sum_7d"] - df["rolling_energy_sum_14d"]
df["energy_release_rate"] = df["rolling_energy_sum_7d"] / 7

energy_mean_30 = df.groupby("grid_id", group_keys=False).apply(
    lambda g: safe_rolling(g, "energy", 30, func="mean"))
energy_std_30 = df.groupby("grid_id", group_keys=False).apply(
    lambda g: safe_rolling(g, "energy", 30, func="std"))
df["energy_anomaly_score"] = (df["energy"] - energy_mean_30) / (energy_std_30 + 1e-6)

for w in [7, 14, 30]:
    df[f"event_count_{w}d"] = df.groupby("grid_id", group_keys=False).apply(
        lambda g: safe_rolling(g, "event_count", w, func="sum"))

df["microquake_density"] = df.groupby("grid_id", group_keys=False).apply(
    lambda g: safe_rolling(g.assign(micro=(g["magnitude"] < 2.5).astype(int)), "micro", 30, func="sum"))

df["foreshock_density"] = df.groupby("grid_id", group_keys=False).apply(
    lambda g: safe_rolling(g.assign(foreshock=((g["magnitude"] >= 3.0) & (g["magnitude"] < 4.0)).astype(int)),
                           "foreshock", 30, func="sum"))

df["cluster_intensity"] = df["event_count_7d"] / (df["event_count_30d"] + 1)
df["seismic_burst_score"] = df["event_count_7d"] / (df["event_count_14d"] + 1)

df["rolling_mean_depth_7d"] = df.groupby("grid_id", group_keys=False).apply(
    lambda g: safe_rolling(g, "depth", 7, func="mean"))

df["rolling_mean_depth_30d"] = df.groupby("grid_id", group_keys=False).apply(
    lambda g: safe_rolling(g, "depth", 30, func="mean"))

df["depth_std_30d"] = df.groupby("grid_id", group_keys=False).apply(
    lambda g: safe_rolling(g, "depth", 30, func="std"))

df["shallow_event_ratio"] = df.groupby("grid_id", group_keys=False).apply(
    lambda g: safe_rolling(g.assign(shallow=(g["depth"] <= 70).astype(int)), "shallow", 30, func="mean"))

df["deep_event_ratio"] = df.groupby("grid_id", group_keys=False).apply(
    lambda g: safe_rolling(g.assign(deep=(g["depth"] >= 300).astype(int)), "deep", 30, func="mean"))

if "country" in df.columns:
    df["regional_event_density"] = df.groupby("country", group_keys=False).apply(
        lambda g: safe_rolling(g, "event_count", 30, func="sum"))
else:
    df["regional_event_density"] = df["event_count_30d"]

df["neighbor_grid_activity"] = df["regional_event_density"]

df['magnitude_depth_ratio'] = df['magnitude'] / (df['depth'] + 1)
df['rolling_mag_mean'] = df.groupby('region')['magnitude'].transform(lambda x: x.rolling(3, min_periods=1).mean())


########################################################################################################################
# STEP 6 : MODELLING (WALK-FORWARD VALIDATION)
########################################################################################################################
print("\n", "STEP 6 : MODELLING", "\n", "=" * 30, "\n")


# Model Preparation
print("\n", "Model Preparation", "\n", "_" * 30, "\n")
drop_cols = ["date", "days_to_next_major_event", "future_max_magnitude_7d", "region", "grid_id"] + cat_cols + cat_but_car
features = [col for col in df.columns if col not in drop_cols]

X = df[features]
y_mag = df['future_max_magnitude_7d']
y_days = df['days_to_next_major_event']


# Walk-Forward Validation (TimeSeriesSplit)
print("\n", "Walk-Forward Validation (TimeSeriesSplit)", "\n", "_" * 30, "\n")
tscv = TimeSeriesSplit(n_splits=5)


# Initialize Models
print("\n", "Initialize Models", "\n", "_" * 30, "\n")
model_mag = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
model_days = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)


# Model Evaluations
print("\n", "Model Evaluations", "\n", "_" * 30, "\n")
print("Evaluating future_max_magnitude_7d Model...")
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_mag.iloc[train_index], y_mag.iloc[test_index]
    model_mag.fit(X_train, y_train)
    preds = model_mag.predict(X_test)
    print(f"(Magnitude) -> MAE: {mean_absolute_error(y_test, preds):.3f} | RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.3f}")

print("\nEvaluating days_to_next_major_event Model...")
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_days.iloc[train_index], y_days.iloc[test_index]
    model_days.fit(X_train, y_train)
    preds = model_days.predict(X_test)
    print(f"(Days) -> MAE: {mean_absolute_error(y_test, preds):.3f} | RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.3f}")


########################################################################################################################
# STEP 7 : FEATURE IMPORTANCE
########################################################################################################################
print("\n", "STEP 7 : FEATURE IMPORTANCE", "\n", "=" * 30, "\n")


# Plot Importance Function
print("\n", "Plot Importance Function", "\n", "_" * 30, "\n")
def plot_importance(model, features_df, target_name, num=10):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features_df.columns})
    plt.figure(figsize=(10, 5))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title(f'Features Importance for {target_name}')
    plt.tight_layout()
    plt.show()


# Train Models
print("\n", "Train Models", "\n", "_" * 30, "\n")
model_mag.fit(X, y_mag)
model_days.fit(X, y_days)


# Plot Importance
print("\n", "Plot Importance", "\n", "_" * 30, "\n")
plot_importance(model_mag, X, "Max Magnitude")
plot_importance(model_days, X, "Days To Event")


########################################################################################################################
# STEP 8 : HYPERPARAMETER OPTIMIZATION
########################################################################################################################
print("\n", "STEP 8 : HYPERPARAMETER OPTIMIZATION", "\n", "=" * 30, "\n")


# Hyperparameter Optimization
# print("\n", "Hyperparameter Optimization", "\n", "_" * 30, "\n")
# rf_params = {"n_estimators": [100, 200, 500], "max_depth": [5, 10, 20, None]}
# grid_search_mag = GridSearchCV(model_mag, rf_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1).fit(X, y_mag)
# grid_search_days = GridSearchCV(model_days, rf_params, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1).fit(X, y_days)
#
# print(f"Best Params (Magnitude): {grid_search_mag.best_params_}")
# print(f"Best Params (Days): {grid_search_days.best_params_}")


# Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_mag(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500, step=100)
    max_depth = trial.suggest_categorical('max_depth', [5, 10, 20, None])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1)

    mse_scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_mag.iloc[train_index], y_mag.iloc[test_index]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, preds))
    return np.mean(mse_scores)

def objective_days(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 500, step=100)
    max_depth = trial.suggest_categorical('max_depth', [5, 10, 20, None])
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1)

    mse_scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_days.iloc[train_index], y_days.iloc[test_index]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse_scores.append(mean_squared_error(y_test, preds))
    return np.mean(mse_scores)

study_mag = optuna.create_study(direction='minimize')
study_mag.optimize(objective_mag, n_trials=30)
print(f"Best Params (Magnitude): {study_mag.best_params}")

study_days = optuna.create_study(direction='minimize')
study_days.optimize(objective_days, n_trials=30)
print(f"Best Params (Days): {study_days.best_params}")


########################################################################################################################
# STEP 9 : FINAL MODEL
########################################################################################################################
print("\n", "STEP 9 : FINAL MODEL", "\n", "=" * 30, "\n")


# Final Models
print("\n", "Final Models", "\n", "_" * 30, "\n")
# final_model_mag = grid_search_mag.best_estimator_
# final_model_days = grid_search_days.best_estimator_
final_model_mag = RandomForestRegressor(**study_mag.best_params, random_state=42, n_jobs=-1)
final_model_days = RandomForestRegressor(**study_days.best_params, random_state=42, n_jobs=-1)

final_model_mag.fit(X, y_mag)
final_model_days.fit(X, y_days)

joblib.dump(final_model_days, 'onur/model_deepfault_days.pkl')
joblib.dump(final_model_mag, 'onur/model_deepfault_mag.pkl')

print("Final Models have been trained with walk-forward validated hyperparameters.")


########################################################################################################################
# STEP 10 : EXPLAINABILITY AND PRODUCT DEVELOPMENT
########################################################################################################################
print("\n", "STEP 10 : EXPLAINABILITY AND PRODUCT DEVELOPMENT (JSON OUTPUT)", "\n", "=" * 30, "\n")


# Generate Prediction JSON
print("\n", "Generate Prediction JSON", "\n", "_" * 30, "\n")
def generate_prediction_json(region_id, current_features):
    mag_tree_preds = np.array([tree.predict(current_features) for tree in final_model_mag.estimators_])
    pred_mag = mag_tree_preds.mean()
    std_dev_mag = mag_tree_preds.std()

    cv_mag = std_dev_mag / pred_mag if pred_mag > 0 else float('inf')

    confidence_mag = 1 / (1 + cv_mag)

    days_tree_preds = np.array([tree.predict(current_features) for tree in final_model_days.estimators_])
    pred_days = days_tree_preds.mean()
    std_dev_days = days_tree_preds.std()

    cv_days = std_dev_days / pred_days if pred_days > 0 else float('inf')
    confidence_days = 1 / (1 + cv_days)

    final_confidence = (confidence_mag + confidence_days) / 2

    output = {
        "region id": region_id,
        "prediction_date": date.today().strftime("%Y-%m-%d"),
        "days to event": round(float(pred_days), 1),
        "predicted max magnitude": round(float(pred_mag), 1),
        "confidence score": round(final_confidence, 2),
        "model_version": "onur_regression_v1"
    }
    return json.dumps(output, indent=4)


def evaluate_cv_metrics(model_params, X, y, splitter):
    mae_scores, rmse_scores = [], []
    for train_index, test_index in splitter.split(X):
        model = RandomForestRegressor(**model_params, random_state=42, n_jobs=-1)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        preds = model.predict(X.iloc[test_index])
        mae_scores.append(mean_absolute_error(y.iloc[test_index], preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y.iloc[test_index], preds)))
    return {
        "mae": float(np.mean(mae_scores)),
        "rmse": float(np.mean(rmse_scores))
    }

def save_model_metadata(output_path="onur/OUTPUTS_onur.json"):
    metrics_mag = evaluate_cv_metrics(study_mag.best_params, X, y_mag, tscv)
    metrics_days = evaluate_cv_metrics(study_days.best_params, X, y_days, tscv)

    metadata = {
        "generated_at": date.today().strftime("%Y-%m-%d"),
        "model_version": "onur_regression_v1",
        "dataset_path": "data/earthquakes_featured.csv",
        "rows": int(df.shape[0]),
        "feature_count": int(len(features)),
        "features": list(features),
        "targets": ["future_max_magnitude_7d", "days_to_next_major_event"],
        "best_params": {
            "future_max_magnitude_7d": study_mag.best_params,
            "days_to_next_major_event": study_days.best_params
        },
        "cv_metrics": {
            "future_max_magnitude_7d": metrics_mag,
            "days_to_next_major_event": metrics_days
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"Model metadata saved to: {output_path}")

def save_predictions_table(output_path="onur/predictions_table_onur.csv"):
    preds_mag = final_model_mag.predict(X)
    preds_days = final_model_days.predict(X)

    base_cols = ["date", "region", "magnitude", "future_max_magnitude_7d", "days_to_next_major_event"]
    existing_cols = [col for col in base_cols if col in df.columns]

    result_df = df[existing_cols].copy()
    result_df["pred_future_max_magnitude_7d"] = preds_mag
    result_df["pred_days_to_next_major_event"] = preds_days

    result_df.to_csv(output_path, index=False)
    print(f"Prediction table saved to: {output_path}")


########################################################################################################################
# STEP 11 : MAIN FUNCTION (EXECUTION)
########################################################################################################################
print("\n", "STEP 11 : MAIN FUNCTION (EXECUTION)", "\n", "=" * 30, "\n")


# Main Function
print("\n", "Main Function", "\n", "_" * 30, "\n")
def main():
    latest_record = X.iloc[[-1]]
    target_region = df['region'].iloc[-1]

    final_json = generate_prediction_json(region_id=target_region, current_features=latest_record)

    print("\n-------------- OUTPUT (JSON) -------------\n")
    print(final_json)
    print("\n------------------------------------------")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    
    output_file_path = os.path.join(script_dir, "onur/prediction_output.json")
    
    with open(output_file_path, "w") as f:
        f.write(final_json)

    save_model_metadata("onur/OUTPUTS_onur.json")
    save_predictions_table("onur/predictions_table_onur.csv")

    return final_json


# Run Function
print("\n", "Run Function", "\n", "_" * 30, "\n")
if __name__ == "__main__":
    main()

print("\n", "-" * 50, "PROJECT FINISHED", "-" * 50, "\n")


