
import joblib
import pandas as pd

# Load the saved model
saved = joblib.load("fraud_model.joblib")

model = saved["model"]
threshold = saved["threshold"]
features = saved["features"]

# Load the dataset
df = pd.read_csv("creditcard.csv")
df = df.drop_duplicates().reset_index(drop=True)

print("\nCREDIT CARD FRAUD DETECTION")
print("-" * 35)

# Select a transaction to test
while True:
    try:
        index = int(input(
            f"Enter a transaction row number (0 to {len(df) - 1}): "
        ))

        if 0 <= index < len(df):
            break

        print("Please enter a valid row number.")

    except ValueError:
        print("Please enter a whole number.")

# Get the transaction and its actual label
transaction = df.iloc[[index]]
X_transaction = transaction[features]
actual_class = int(transaction["Class"].iloc[0])

# Predict fraud probability
fraud_probability = model.predict_proba(X_transaction)[0, 1]
prediction = int(fraud_probability >= threshold)

# Display results
print("\n--- RESULTS ---")
print(f"Transaction row: {index}")
print(f"Fraud probability: {fraud_probability:.2%}")
print(
    "Prediction:",
    "FRAUD" if prediction == 1 else "LEGITIMATE"
)
print(
    "Actual class:",
    "FRAUD" if actual_class == 1 else "LEGITIMATE"
)
print(
    "Correct prediction:",
    "YES" if prediction == actual_class else "NO"
)
