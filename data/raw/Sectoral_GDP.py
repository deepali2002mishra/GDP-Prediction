import os
import requests
import pandas as pd
import yfinance as yf

# Define save directory
SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🌎 World Bank API Base URL
WB_API_URL = "https://api.worldbank.org/v2/country/IND/indicator/{}?date=2000:2024&format=json"

# 🔹 Sector-Specific Economic Indicators with API Sources
SECTOR_INDICATORS = {
    "Agriculture": {
        "Agricultural GDP (% of GDP)": "NV.AGR.TOTL.ZS",  # World Bank
        "Fertilizer Consumption (kg per hectare)": "AG.CON.FERT.ZS",  # World Bank
        "Crop Yield (kg per hectare)": "AG.YLD.CREL.KG",  # FAO API
        "Rural Employment Rate (%)": "SL.AGR.EMPL.ZS",  # World Bank
        "Agricultural Exports (Billion USD)": "TX.VAL.AGRI.ZS.UN",  # FAO API
    },
    "Services": {
        "Services GDP (% of GDP)": "NV.SRV.TOTL.ZS",  # World Bank
        "IT & Software Exports (Billion USD)": "BX.GSR.CCIS.CD",  # World Bank
        "Tourism Revenue (Billion USD)": "ST.INT.RCPT.CD",  # World Bank
        "Telecom Subscribers Growth (%)": "https://api.trai.gov.in/telecom_data",  # TRAI API
        "Consumer Confidence Index (CCI)": "https://stats.oecd.org/SDMX-JSON/data/DP_LIVE/IND.CCI.TOT.A/OECD?contentType=csv",  # OECD API
    },
    "Real_Estate": {
        "Real Estate GDP (% of GDP)": "https://rbi.gov.in/api/real_estate_gdp",  # RBI API
        "Housing Price Index (HPI)": "https://rbi.gov.in/api/housing_price_index",  # RBI API
        "Bank Lending to Real Estate (Billion USD)": "https://rbi.gov.in/api/bank_lending_real_estate",  # RBI API
        "Cement & Steel Production (Million Tons)": "https://industrydata.gov.in/api/construction_materials",  # Indian Government API
        "Urban Population Growth (%)": "SP.URB.GROW",  # World Bank
    },
    "Energy": {
        "Energy GDP (% of GDP)": "https://mnre.gov.in/api/energy_gdp",  # Ministry of Renewable Energy API
        "Oil & Gas Production (Million Barrels)": "https://petroleum.gov.in/api/oil_gas_production",  # Indian Petroleum Ministry API
        "Renewable Energy Investments (Billion USD)": "https://mnre.gov.in/api/renewable_investments",  # MNRE API
        "Electricity Demand Growth (%)": "https://cea.gov.in/api/electricity_demand",  # Central Electricity Authority API
    },
    "Automobile": {
        "Automobile Sector GDP (% of GDP)": "https://siamindia.com/api/automobile_gdp",  # SIAM API
        "Total Vehicle Sales (Million Units)": "https://siamindia.com/api/vehicle_sales",  # SIAM API
        "Fuel Prices (INR per Liter)": "https://petroleum.gov.in/api/fuel_prices",  # Petroleum Ministry API
        "Freight Traffic (Million Ton-Km)": "https://indianrailways.gov.in/api/freight_traffic",  # Indian Railways API
        "EV Adoption Rate (%)": "https://mnre.gov.in/api/ev_adoption",  # MNRE API
    }
}

# Function to fetch data from World Bank API with error handling
def fetch_world_bank_data(indicator):
    """Fetches data from World Bank API for a given indicator with error handling."""
    url = WB_API_URL.format(indicator)
    response = requests.get(url)
    
    try:
        data = response.json()
        
        # Check if response contains valid data
        if isinstance(data, list) and len(data) > 1 and isinstance(data[1], list):
            records = [{"Year": int(item["date"]), "Value": item["value"]} for item in data[1] if item["value"] is not None]
            return pd.DataFrame(records)
        else:
            print(f"⚠️ Warning: No valid data found for indicator {indicator}. Response: {data}")
            return pd.DataFrame(columns=["Year", "Value"])
    
    except Exception as e:
        print(f"❌ Error fetching data for indicator {indicator}: {e}")
        return pd.DataFrame(columns=["Year", "Value"])


# Function to fetch data from other REST APIs
def fetch_rest_api_data(url):
    """Fetches data from REST API endpoints."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error fetching data from {url}: {e}")
        return pd.DataFrame()

# Function to fetch stock/financial data from Yahoo Finance
def fetch_stock_data(ticker):
    """Fetches stock data from Yahoo Finance for the given ticker."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max", start="2000-01-01", end="2024-12-31")
    hist = hist.resample('YE').last()
    hist['Year'] = hist.index.year.astype(int)
    return hist[['Year', 'Close']].rename(columns={"Close": f"{ticker} Close Price"})

# Function to extract and save data for each sector
def extract_and_save_sector_data():
    """Extracts economic data for each sector and saves as CSV files."""
    for sector, indicators in SECTOR_INDICATORS.items():
        print(f"Fetching data for {sector} sector...")

        sector_data = pd.DataFrame()

        for indicator_name, indicator_code in indicators.items():
            if indicator_code:
                if "worldbank.org" in WB_API_URL:
                    df = fetch_world_bank_data(indicator_code)
                elif "yahoo" in indicator_code:
                    df = fetch_stock_data(indicator_code)
                elif "api" in indicator_code:
                    df = fetch_rest_api_data(indicator_code)
                else:
                    continue  # Skip missing or unknown indicators

                if not df.empty:
                    df = df.rename(columns={"Value": indicator_name})
                    if sector_data.empty:
                        sector_data = df
                    else:
                        sector_data = sector_data.merge(df, on="Year", how="outer")

        # Save sector data as CSV
        if not sector_data.empty:
            file_path = os.path.join(SAVE_DIR, f"{sector}_data.csv")
            sector_data.to_csv(file_path, index=False)
            print(f"✅ {sector} data saved to {file_path}")
        else:
            print(f"⚠️ No data found for {sector}")

# Run the data extraction process
if __name__ == "__main__":
    extract_and_save_sector_data()
