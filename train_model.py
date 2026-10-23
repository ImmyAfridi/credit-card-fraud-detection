
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    confusion_matrix,
)

# 1. Load the dataset
df = pd.read_csv("creditcard.csv")

# 2. Remove exact duplicate rows
df_clean = df.drop_duplicates().copy()

# 3. Separate features and target
X = df_clean.drop(columns="Class")
y = df_clean["Class"]

# 4. Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 5. Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

# 6. Evaluate the model
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.4
y_pred = (y_prob >= threshold).astype(int)

print(f"Decision threshold: {threshold}")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraud"],
        zero_division=0,
    )
)

print(
    "Average Precision:",
    round(average_precision_score(y_test, y_prob), 4),
)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("\nConfusion Matrix Counts:")
print(f"True Negatives:  {tn:,}")
print(f"False Positives: {fp:,}")
print(f"False Negatives: {fn:,}")
print(f"True Positives:  {tp:,}")

# 7. Save the trained model and threshold
joblib.dump(
    {
        "model": model,
        "threshold": threshold,
        "features": list(X.columns),
    },
    "fraud_model.joblib",
)

print("\nModel saved as fraud_model.joblib")
