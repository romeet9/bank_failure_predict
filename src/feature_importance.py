import pandas as pd
import joblib

# Load dataset
file_path = "/Users/romeet/Desktop/Projects/bank_failure_predict/data/processed/cleaned_data.xlsx"
df = pd.read_excel(file_path)

# Define target and features
X = df.drop(columns=["bank_failure_risk"])
y = df["bank_failure_risk"]

# Load trained models
rf_model = joblib.load(
    "/Users/romeet/Desktop/Projects/bank_failure_predict/models/Random_Forest.pkl"
)
xgb_model = joblib.load(
    "/Users/romeet/Desktop/Projects/bank_failure_predict/models/Gradient_Boosting_(XGBoost).pkl"
)

# Get feature importance scores
rf_importance = rf_model.feature_importances_
xgb_importance = xgb_model.feature_importances_

# Create DataFrame for feature importance
importance_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Random Forest Importance": rf_importance,
        "XGBoost Importance": xgb_importance,
    }
).sort_values(by="Random Forest Importance", ascending=False)

# Save feature importance
importance_df.to_excel(
    "/Users/romeet/Desktop/Projects/bank_failure_predict/reports/feature_importance.xlsx",
    index=False,
)

# Select top 5 features dynamically
top_features = importance_df["Feature"].head(5).tolist()

# Save the final dataset with top features
df_final = df[top_features + ["bank_failure_risk"]]
df_final.to_excel(
    "/Users/romeet/Desktop/Projects/bank_failure_predict/data/processed/final_dataset.xlsx",
    index=False,
)
print(f"✅ Feature selection complete! Using top {len(top_features)} features.")
