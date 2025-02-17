import pandas as pd

# Load dataset
file_path = "/Users/romeet/Desktop/Projects/bank_failure_predict/data/raw/project4_data_bank_failure.xlsx"
df = pd.read_excel(file_path)

# Define threshold as the 80th percentile of bank failure ratio
threshold = df["Failed_to_Active_banks_ratio"].quantile(0.80)

# Convert to binary classification
df["bank_failure_risk"] = (df["Failed_to_Active_banks_ratio"] > threshold).astype(int)

# Drop the original ratio column
df.drop(columns=["Failed_to_Active_banks_ratio"], inplace=True)

# Drop unnecessary columns (e.g., 'gdp' is dropped since we use 'ld_gdp')
columns_to_drop = ["gdp", "obs"]
df_cleaned = df.drop(columns=columns_to_drop, errors="ignore")

# Handle missing values
df_cleaned.fillna(method="ffill", inplace=True)
df_cleaned.dropna(inplace=True)

# Save cleaned data
df_cleaned.to_excel(
    "/Users/romeet/Desktop/Projects/bank_failure_predict/data/processed/cleaned_data.xlsx",
    index=False,
)
print("✅ Cleaned dataset saved as 'cleaned_data.xlsx'.")

# Define strong and weak predictors
strong_predictors = [
    "Loan Loss Reserve to Total Loans for all U.S. Banks; Percent",
    "Net Interest Margin for all Commerical Banks",
    "Return on Equity ",
]

weak_predictors = [
    "sp500_dev",
    "XMRET",
    "Allowance for Loan and Lease Losses , Percent Change, Quarterly",
    "EBP",
    "ld_gdp",
    "d_fedFunds_rate",
    "TERM_SPREAD",
]

# Print strong and weak predictors
print("\n✅ Strong Predictors (Kept):")
for feature in strong_predictors:
    print(f"- {feature}")

print("\n❌ Weak Predictors (Dropped):")
for feature in weak_predictors:
    print(f"- {feature}")

# Keep strong predictors + target variable
df_selected = df[strong_predictors + ["bank_failure_risk"]]

# Handle missing values
df_selected.fillna(method="ffill", inplace=True)
df_selected.dropna(inplace=True)

# Save processed dataset
df_selected.to_excel(
    "/Users/romeet/Desktop/Projects/bank_failure_predict/data/processed/final_dataset.xlsx",
    index=False,
)

print("\n✅ Feature selection complete! 'final_dataset.xlsx' saved correctly.")

# Print column names to verify
print("\n📌 Columns in the final dataset:")
for col in df_selected.columns:
    print(f"- {col}")
