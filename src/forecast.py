import pandas as pd
import pickle
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA

# Define Paths
DATA_FILE = r"D:\Projects\GDP_Prediction_Project\data\processed\feature_engineered.csv"
ARIMA_MODEL_PATH = r"D:\Projects\GDP_Prediction_Project\models\arima_model.pkl"
XGB_MODEL_PATH = r"D:\Projects\GDP_Prediction_Project\models\xgboost_model.pkl"
RESULTS_FILE = r"D:\Projects\GDP_Prediction_Project\results\gdp_forecast.csv"

# Ensure results directory exists
RESULTS_DIR = os.path.dirname(RESULTS_FILE)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_models():
    """Loads the trained ARIMA and XGBoost models."""
    print("📥 Loading ARIMA model...")
    arima_model = pickle.load(open(ARIMA_MODEL_PATH, "rb"))
    
    print("📥 Loading XGBoost model...")
    xgb_model = pickle.load(open(XGB_MODEL_PATH, "rb"))
    
    return arima_model, xgb_model

def forecast_arima(model, df, steps=5):
    """Generates GDP forecasts using the ARIMA model."""
    print("📈 Forecasting GDP using ARIMA model...")

    # Fix: Ensure year is an integer before converting to string
    last_year = int(df["Year"].max())  
    future_years = pd.date_range(start=str(last_year + 1), periods=steps, freq="YE").year
    
    forecast = model.predict(start=len(df), end=len(df) + steps - 1)
    return pd.DataFrame({"Year": future_years, "GDP Growth (%) (ARIMA)": forecast})

def forecast_xgboost(model, df, steps=5):
    """Generates GDP forecasts using the XGBoost model."""
    print("📈 Forecasting GDP using XGBoost model...")
    
    latest_data = df.drop(columns=['GDP Growth (%)', 'Year'], errors='ignore').iloc[-1:].values
    predictions = [model.predict(latest_data)[0]]

    for _ in range(steps - 1):
        latest_data = np.roll(latest_data, -1)  
        predictions.append(model.predict(latest_data)[0])
    
    last_year = int(df["Year"].max())  
    future_years = pd.date_range(start=str(last_year + 1), periods=steps, freq="YE").year
    
    return pd.DataFrame({"Year": future_years, "GDP Growth (%) (XGBoost)": predictions})

def forecast_hybrid(arima_forecast, xgb_forecast):
    """Combines ARIMA and XGBoost forecasts into a hybrid prediction."""
    print("⚡ Combining ARIMA and XGBoost predictions into a Hybrid Model...")
    
    hybrid_forecast = arima_forecast.merge(xgb_forecast, on="Year", how="left")
    hybrid_forecast["GDP Growth (%) (Hybrid)"] = (
        0.5 * hybrid_forecast["GDP Growth (%) (ARIMA)"] + 
        0.5 * hybrid_forecast["GDP Growth (%) (XGBoost)"]
    )
    
    return hybrid_forecast

if __name__ == "__main__":
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Load trained models
    arima_model, xgb_model = load_models()

    # Forecast GDP using ARIMA and XGBoost
    arima_forecast = forecast_arima(arima_model, df)
    xgb_forecast = forecast_xgboost(xgb_model, df)

    # Create Hybrid Forecast
    hybrid_forecast = forecast_hybrid(arima_forecast, xgb_forecast)

    # Save Forecast
    hybrid_forecast.to_csv(RESULTS_FILE, index=False)
    print(f"✅ Forecast saved to: {RESULTS_FILE}")
