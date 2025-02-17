import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

# Load dataset
file_path = "/Users/romeet/Desktop/Projects/bank_failure_predict/data/processed/final_dataset.xlsx"
df = pd.read_excel(file_path)

# Define target and features
X = df.drop(columns=["bank_failure_risk"])
y = df["bank_failure_risk"]

# Split data into training (80%) & testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features (important for SVM, MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(
    scaler, "/Users/romeet/Desktop/Projects/bank_failure_predict/models/scaler.pkl"
)

# Initialize models with MLP fix
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting (XGBoost)": XGBClassifier(
        eval_metric="logloss", random_state=42
    ),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(64, 32),  # Two hidden layers
        max_iter=2000,  # Increased max iterations
        learning_rate_init=0.0005,  # Lower learning rate
        early_stopping=True,  # Stop if validation loss stops improving
        random_state=42,
    ),
    "Linear Regression (OLS)": LinearRegression(),
}

# Train models and evaluate performance
auroc_scores = {}

for name, model in models.items():
    print(f"Training {name}...")

    # Use standardized data for SVM, MLP
    if name in ["SVM", "Neural Network (MLP)"]:
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.predict(X_test)
        )

    # Compute AUROC Score
    auroc = roc_auc_score(y_test, y_pred_proba)
    auroc_scores[name] = auroc

    # Save trained models
    joblib.dump(
        model,
        f"/Users/romeet/Desktop/Projects/bank_failure_predict/models/{name.replace(' ', '_')}.pkl",
    )

# Print AUROC Scores
print("\n📊 Model AUROC Scores:")
for model, score in auroc_scores.items():
    print(f"{model}: {score:.4f}")

# Save scores for reporting
pd.DataFrame(auroc_scores.items(), columns=["Model", "AUROC"]).to_excel(
    "/Users/romeet/Desktop/Projects/bank_failure_predict/reports/model_performance.xlsx",
    index=False,
)

print("\n✅ Model training complete! Trained models saved in 'models/'")
