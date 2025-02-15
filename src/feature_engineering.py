import pandas as pd
import numpy as np
import os

# Define dataset paths
INPUT_FILE = r"D:\Projects\GDP_Prediction_Project\data\processed\cleaned_data.csv"
OUTPUT_FILE = r"D:\Projects\GDP_Prediction_Project\data\processed\feature_engineered.csv"

def create_lag_features(df, columns, lags):
    """Creates lag-based features for time-series modeling."""
    lag_dfs = []
    for lag in lags:
        lag_df = df[columns].shift(lag).add_suffix(f"_lag{lag}")
        lag_dfs.append(lag_df)
    return pd.concat([df] + lag_dfs, axis=1)

def create_rolling_features(df, columns, windows):
    """Creates rolling mean and standard deviation features."""
    roll_dfs = []
    for window in windows:
        roll_mean_df = df[columns].rolling(window=window, min_periods=1).mean().add_suffix(f"_roll_mean{window}")
        roll_std_df = df[columns].rolling(window=window, min_periods=1).std().add_suffix(f"_roll_std{window}")
        roll_dfs.extend([roll_mean_df, roll_std_df])
    return pd.concat([df] + roll_dfs, axis=1)

def create_growth_rate_features(df, columns):
    """Computes percentage change for economic indicators."""
    growth_df = df[columns].pct_change().multiply(100).add_suffix("_growth")
    df = pd.concat([df, growth_df], axis=1)
    df.fillna(0, inplace=True)
    return df

def create_interaction_features(df):
    """Creates economic ratios that provide meaningful insights."""
    df["FDI_to_GDP"] = df["FDI (Billion USD)"] / df["GDP Growth (%)"]
    df["Exports_to_Imports"] = df["Exports (Billion USD)"] / df["Imports (Billion USD)"]
    df["MoneySupply_to_GDP"] = df["Money Supply (M3) Growth (%)"] / df["GDP Growth (%)"]
    df.fillna(0, inplace=True)
    return df

def create_cyclical_features(df):
    """Creates cyclical time-based features from the 'Year' column."""
    df["Year_sin"] = np.sin(2 * np.pi * df["Year"] / df["Year"].max())
    df["Year_cos"] = np.cos(2 * np.pi * df["Year"] / df["Year"].max())
    return df

if __name__ == "__main__":
    df = pd.read_csv(INPUT_FILE)

    # List of economic indicators for feature engineering
    indicators = [
        'GDP Growth (%)', 'Inflation Rate (%)', 'Interest Rate (%)',
        'Exchange Rate (USD/INR)', 'Fiscal Deficit (% of GDP)',
        'Exports (Billion USD)', 'Imports (Billion USD)', 'FDI (Billion USD)',
        'Money Supply (M3) Growth (%)', 'Bank Credit Growth (%)',
        'Unemployment Rate (%)', 'Private Consumption (% of GDP)',
        'Fixed Capital Formation (% of GDP)', 'Trade Balance (Billion USD)',
        '^NSEI Close Price', '^BSESN Close Price', 'CCI', 'Manufacturing PMI'
    ]

    print(f"Original data rows: {df.shape[0]}")

    df = create_lag_features(df, indicators, lags=[1, 3, 6, 12])
    df = create_rolling_features(df, indicators, windows=[3, 6, 12])
    df = create_growth_rate_features(df, indicators)
    df = create_interaction_features(df)
    df = create_cyclical_features(df)

    df.dropna(inplace=True)  # Drop NaN values only at the end
    print(f"Final rows after feature engineering: {df.shape[0]}")

    # Fix: Ensure the file is closed before writing
    if os.path.exists(OUTPUT_FILE):
        try:
            os.remove(OUTPUT_FILE)  # Delete the existing file
        except PermissionError:
            print(f"⚠️ Warning: Cannot delete {OUTPUT_FILE}. Close the file and retry.")
            exit(1)

    # Save feature-engineered dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print("✅ Feature Engineering completed. Data saved at:", OUTPUT_FILE)
