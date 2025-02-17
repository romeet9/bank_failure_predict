import os

print("🚀 Starting the full pipeline...\n")

# Step 1: Data Cleaning
print("🔹 Running data processing...")
os.system("python src/data_processing.py")

# Step 2: Initial Model Training
print("🔹 Training initial models...")
os.system("python src/model_training.py")

# Step 3: Feature Importance Analysis
print("🔹 Analyzing feature importance...")
os.system("python src/feature_importance.py")

# # Step 4: Retrain models with top features
# print("🔹 Retraining models with selected features...")
# os.system("python src/model_training.py")

print("\n✅ Full pipeline execution complete! Check reports for results.")
