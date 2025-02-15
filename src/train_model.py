import pandas as pd
import pickle
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Define dataset paths
INPUT_FILE = r"D:\Projects\GDP_Prediction_Project\data\processed\feature_engineered.csv"
ARIMA_MODEL_PATH = r"D:\Projects\GDP_Prediction_Project\models\arima_model.pkl"
XGB_MODEL_PATH = r"D:\Projects\GDP_Prediction_Project\models\xgboost_model.pkl"
HYBRID_MODEL_PATH = r"D:\Projects\GDP_Prediction_Project\models\hybrid_model.pkl"

def clean_data(df):
    """Cleans the dataset by handling NaN, Inf values, and forward-filling missing data."""
    df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Convert inf to NaN
    missing_before = df.isnull().sum().sum()
    
    df.ffill(inplace=True)  # Forward-fill missing values
    df.fillna(0, inplace=True)  # Replace remaining NaNs with 0
    
    missing_after = df.isnull().sum().sum()
    print(f"🔍 Cleaned dataset: Removed {missing_before - missing_after} missing values.")
    return df

def train_arima(df):
    """Trains an ARIMA model on GDP Growth (%) with proper time indexing."""
    df = df.set_index("Year")  # Ensure Year is the index
    df.index = pd.to_datetime(df.index, format="%Y").to_period("Y")  # Set yearly frequency

    print("🚀 Training ARIMA model...")
    model = ARIMA(df['GDP Growth (%)'], order=(5,1,0))  # ARIMA(5,1,0)
    model_fit = model.fit()
    return model_fit

def train_xgboost(df):
    """Trains an XGBoost regression model for GDP forecasting."""
    X = df.drop(columns=['GDP Growth (%)', 'Year'], errors='ignore')  # Drop target & Year column
    y = df['GDP Growth (%)']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print dataset summary for debugging
    print("🔹 XGBoost Training Data Summary 🔹")
    print(f"Total Features: {X_train.shape[1]}, Training Samples: {X_train.shape[0]}")
    print(X_train.describe().T)  # Show feature statistics
    
    # Train XGBoost Model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    return model

def train_hybrid_model(arima_model, xgb_model, df):
    """Creates a hybrid model by combining ARIMA and XGBoost predictions."""
    print("⚡ Training Hybrid Model...")
    
    # Predict using ARIMA
    arima_pred = arima_model.predict(start=df.index[0], end=df.index[-1])
    
    # Predict using XGBoost
    X = df.drop(columns=['GDP Growth (%)', 'Year'], errors='ignore')
    xgb_pred = xgb_model.predict(X)
    
    # Combine predictions (Weighted Average)
    hybrid_pred = (0.5 * arima_pred) + (0.5 * xgb_pred)
    
    print("✅ Hybrid Model trained successfully.")
    return hybrid_pred

if __name__ == "__main__":
    print("📂 Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    print(f"✅ Dataset Loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    # Clean Data
    df = clean_data(df)

    # Train & Save ARIMA Model
    arima_model = train_arima(df)
    pickle.dump(arima_model, open(ARIMA_MODEL_PATH, "wb"))
    print("✅ ARIMA model saved.")

    # Train & Save XGBoost Model
    xgb_model = train_xgboost(df)
    pickle.dump(xgb_model, open(XGB_MODEL_PATH, "wb"))
    print("✅ XGBoost model saved.")

    # Train & Save Hybrid Model
    hybrid_predictions = train_hybrid_model(arima_model, xgb_model, df)
    pickle.dump(hybrid_predictions, open(HYBRID_MODEL_PATH, "wb"))
    print("✅ Hybrid model predictions saved.")