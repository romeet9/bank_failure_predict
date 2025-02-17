import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "/Users/romeet/Desktop/Projects/bank_failure_predict/data/processed/final_dataset.xlsx"
df = pd.read_excel(file_path)

# Plot: Bank Failure Over Time
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x=df.index, y="bank_failure_risk", marker="o")
plt.title("Bank Failures Over Time")
plt.xlabel("Index (Time Series)")
plt.ylabel("Bank Failure (1=High Risk, 0=Low Risk)")
plt.savefig(
    "/Users/romeet/Desktop/Projects/bank_failure_predict/reports/bank_failures_over_time.png"
)
plt.show()

# Plot: Feature Importance (Bar Chart)
feature_importance = pd.read_csv(
    "/Users/romeet/Desktop/Projects/bank_failure_predict/reports/feature_importance.csv"
).set_index("Feature")
plt.figure(figsize=(10, 5))
feature_importance.plot(kind="bar")
plt.title("Feature Importance (Random Forest & XGBoost)")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.savefig(
    "/Users/romeet/Desktop/Projects/bank_failure_predict/reports/feature_importance.png"
)
plt.show()

# Plot: Heatmap of Feature Correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig(
    "/Users/romeet/Desktop/Projects/bank_failure_predict/reports/feature_correlation_heatmap.png"
)
plt.show()

print("✅ All visualizations saved in 'reports/' folder.")
