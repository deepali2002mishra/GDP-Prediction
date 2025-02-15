import pandas as pd

# Define file path
file_path = r"C:\Users\Deepali Mishra\Downloads\export-2025-02-12T17_18_28.660Z.csv"

# Read the CSV while skipping the first three lines
cci_df = pd.read_csv(file_path, skiprows=3, header=None, names=["Date", "CCI"])

# Convert 'Date' column to datetime format
cci_df["Date"] = pd.to_datetime(cci_df["Date"])

# Extract the Year
cci_df["Year"] = cci_df["Date"].dt.year

# Compute the annual average CCI for each year
cci_annual_df = cci_df.groupby("Year")["CCI"].mean().reset_index()

# Save the processed data to a new CSV file
cleaned_csv_path = "cci_annual_data.csv"
cci_annual_df.to_csv(cleaned_csv_path, index=False)

# Print the processed data instead of using ace_tools
print(cci_annual_df.head())
print(f"✅ Annual CCI data saved as {cleaned_csv_path}")
