import requests
import pandas as pd
import yfinance as yf

# Function to fetch data from World Bank API
def fetch_world_bank_data(indicator, country="IND", start_year=2000, end_year=2024):
    url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start_year}:{end_year}&format=json"
    response = requests.get(url)
    data = response.json()
    
    if len(data) > 1 and isinstance(data[1], list):
        records = [{"Year": int(item["date"]), "Value": item["value"]} for item in data[1] if item["value"] is not None]
        df = pd.DataFrame(records)
        return df.sort_values("Year", ascending=True)
    else:
        return pd.DataFrame(columns=["Year", "Value"])

# World Bank Indicators Mapping
indicators = {
    "GDP Growth (%)": "NY.GDP.MKTP.KD.ZG",
    "Inflation Rate (%)": "FP.CPI.TOTL.ZG",
    "Interest Rate (%)": "FR.INR.RINR",
    "Exchange Rate (USD/INR)": "PA.NUS.FCRF",
    "Fiscal Deficit (% of GDP)": "GC.BAL.CASH.GD.ZS",
    "Exports (Billion USD)": "NE.EXP.GNFS.CD",
    "Imports (Billion USD)": "NE.IMP.GNFS.CD",
    "Trade Balance (Billion USD)": None,  # Will be calculated as Exports - Imports
    "FDI (Billion USD)": "BX.KLT.DINV.CD.WD",
    "Money Supply (M3) Growth (%)": "FM.LBL.MQMY.ZG",
    "Bank Credit Growth (%)": "FS.AST.PRVT.GD.ZS",
    "Unemployment Rate (%)": "SL.UEM.TOTL.ZS",
    "Private Consumption (% of GDP)": "NE.CON.PETC.ZS",
    "Fixed Capital Formation (% of GDP)": "NE.GDI.FTOT.ZS",
}

# Fetch data from World Bank API
data_frames = {}
for key, indicator in indicators.items():
    if indicator:
        df = fetch_world_bank_data(indicator)
        data_frames[key] = df.set_index("Year")

# Flatten multi-index and reset index
final_df = pd.concat(data_frames, axis=1)  # Keep original column names
final_df.columns = [col[0] for col in final_df.columns]  # Flatten multi-index
final_df = final_df.reset_index()  # Ensure "Year" is a proper column

# Handling Trade Balance (Exports - Imports)
if "Exports (Billion USD)" in final_df.columns and "Imports (Billion USD)" in final_df.columns:
    final_df["Trade Balance (Billion USD)"] = final_df["Exports (Billion USD)"] - final_df["Imports (Billion USD)"]

# Function to fetch stock data from Yahoo Finance (NIFTY & SENSEX)
def fetch_stock_data(ticker, start_year="2000-01-01", end_year="2024-12-31"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max", start=start_year, end=end_year)
    hist = hist.resample('YE').last()  # Fix deprecation warning for 'Y'
    hist['Year'] = hist.index.year.astype(int)  # Ensure 'Year' is integer
    return hist[['Year', 'Close']].rename(columns={"Close": f"{ticker} Close Price"})

# Fetch NIFTY 50 & SENSEX Data
nifty_df = fetch_stock_data("^NSEI")
sensex_df = fetch_stock_data("^BSESN")

# Merge stock data
final_df = final_df.merge(nifty_df, on="Year", how="left")
final_df = final_df.merge(sensex_df, on="Year", how="left")

# Function to fetch Consumer Confidence Index (CCI) from OECD API
def fetch_cci_data():
    url = "https://stats.oecd.org/SDMX-JSON/data/DP_LIVE/IND.CCI.TOT.AGRWTH.A/OECD?contentType=csv"
    try:
        cci_df = pd.read_csv(url)
        cci_df = cci_df[['TIME', 'Value']]
        cci_df.columns = ['Year', 'Consumer Confidence Index (CCI)']
        cci_df['Year'] = cci_df['Year'].astype(int)
        return cci_df
    except Exception as e:
        print(f"⚠️ Failed to fetch CCI data: {e}")
        return pd.DataFrame(columns=["Year", "Consumer Confidence Index (CCI)"])

# Fetch Consumer Confidence Index (CCI) Data
cci_df = fetch_cci_data()
if not cci_df.empty:
    final_df = final_df.merge(cci_df, on="Year", how="left")

# Save to CSV
csv_filename = "national_economic_indicators.csv"
final_df.to_csv(csv_filename, index=False)
print(f"✅ Data successfully saved to {csv_filename}")
