import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define dataset paths
DATASET_PATH = r"D:\Projects\GDP_Prediction_Project\national_economic_indicators_1980_2024.csv"
OUTPUT_PATH = r"D:\Projects\GDP_Prediction_Project\data\processed\cleaned_data.csv"

def load_data(file_path):
    """Loads dataset from a CSV file and ensures correct column names."""
    df = pd.read_csv(file_path).copy()  # Ensure deep copy to avoid warnings
    df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
    print("✅ Columns in dataset:", df.columns.tolist())  # Debugging
    return df

def clean_data(df):
    """Handles missing values and ensures data consistency."""
    df = df.copy()  # Ensure we're working on a new copy

    # Ensure "Year" is numeric and drop rows where "Year" is missing
    df.loc[:, "Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"]).copy()  # Drop only where Year is NaN
    df.loc[:, "Year"] = df["Year"].astype(int)

    # Remove '%' symbols and convert percentage-based columns to float
    percentage_columns = [
        'GDP Growth (%)', 'Inflation Rate (%)', 'Money Supply (M3) Growth (%)',
        'Bank Credit Growth (%)', 'Fiscal Deficit (% of GDP)', 'Private Consumption (% of GDP)',
        'Fixed Capital Formation (% of GDP)', 'Unemployment Rate (%)'
    ]

    for col in percentage_columns:
        if col in df.columns:
            df.loc[:, col] = df[col].astype(str).str.replace('%', '', regex=True).astype(float)

    # Sort data by Year before filling missing values
    df = df.sort_values("Year").copy()

    # Prevent FutureWarning by converting object columns before filling missing values
    df = df.infer_objects(copy=False)

    # Forward-fill and backward-fill missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Explicitly re-infer data types after filling
    df = df.infer_objects(copy=False)

    # Prevent future Pandas automatic downcasting warnings
    pd.set_option('future.no_silent_downcasting', True)

    return df

def scale_features(df, features):
    """Scales specified features using MinMaxScaler."""
    scaler = MinMaxScaler()

    # Convert all feature columns to numeric (fix non-numeric errors)
    for col in features:
        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

    # Fill any remaining NaNs with column means (last resort)
    df.fillna(df.mean(), inplace=True)

    # Scale numeric features
    df[features] = scaler.fit_transform(df[features])

    return df, scaler

if __name__ == "__main__":
    # Load and clean data
    df = load_data(DATASET_PATH)
    df = clean_data(df)
    
    # Define the features to scale
    features_to_scale = [
        'Inflation Rate (%)', 'Interest Rate (%)', 'Exchange Rate (USD/INR)',
        'Fiscal Deficit (% of GDP)', 'Exports (Billion USD)', 'Imports (Billion USD)',
        'FDI (Billion USD)', 'Money Supply (M3) Growth (%)', 'Bank Credit Growth (%)',
        'Unemployment Rate (%)', 'Private Consumption (% of GDP)', 'Fixed Capital Formation (% of GDP)',
        'Trade Balance (Billion USD)', '^NSEI Close Price', '^BSESN Close Price',
        'CCI', 'Manufacturing PMI'
    ]

    # Ensure selected features exist in the dataset
    features_to_scale = [col for col in features_to_scale if col in df.columns]

    # Scale features
    df, scaler = scale_features(df, features_to_scale)

    # Save cleaned data
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Data preprocessing completed. Cleaned data saved at: {OUTPUT_PATH}")
