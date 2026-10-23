# Credit Card Fraud Detection

## Overview
This project uses machine learning to identify potentially fraudulent credit card transactions. It explores severe class imbalance and evaluates models using metrics suited to fraud detection.

## Dataset
- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features:** 30 input features, including `Time`, `Amount`, and anonymized features `V1`–`V28`
- **Target:** `Class` — `0` for legitimate transactions and `1` for fraudulent transactions
- **Challenge:** Fraud represents a very small proportion of the transactions.

The dataset is not included in this repository because of its size. Download it using the provided script or from Kaggle.

## Models and Approach
- Explored the data and examined class imbalance and duplicate rows.
- Removed exact duplicate records for the modeling experiment.
- Used a stratified train/test split.
- Established Logistic Regression as a baseline.
- Trained a Random Forest classifier with class weighting.
- Compared decision thresholds using a validation set.
- Selected a threshold of `0.4` based on validation F1-score.

## Final Test Results
The final validation-informed Random Forest workflow achieved the following results on a held-out test set:

| Metric | Result |
|---|---:|
| Fraud precision | 92% |
| Fraud recall | 73.7% |
| Fraud F1-score | 0.82 |
| Average Precision | 0.8115 |
| True positives | 70 |
| False positives | 6 |
| True negatives | 56,645 |
| False negatives | 25 |

The final metrics above came from the notebook workflow. Running `train_model.py` independently creates a new split, so its results may differ.

## Project Structure
```text
credit-card-fraud-detection/
├── fraud_detection.ipynb
├── explore_data.py
├── download_dataset.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/ImmyAfridi/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   ```
   On Windows:
   ```bash
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from Kaggle or use `download_dataset.py`. Ensure `creditcard.csv` is in the project folder.

## Train the Model
```bash
python train_model.py
```

This trains a Random Forest model, prints evaluation metrics, and saves `fraud_model.joblib`.

## Make a Prediction
```bash
python predict.py
```

Follow the prompt to select a transaction row from the dataset. The script displays the predicted class, fraud probability, and actual class for comparison.

## Limitations
- The dataset is highly imbalanced, so accuracy alone is not a sufficient evaluation metric.
- A prediction threshold of `0.4` reflects this experiment's validation results and may not be suitable for every real-world fraud detection system.
- This is an educational portfolio project, not a production fraud detection service.
- The anonymized features limit direct interpretation of why individual transactions are flagged.

## Technologies
Python · pandas · scikit-learn · Matplotlib · Seaborn · Jupyter Notebook · Git · GitHub

## Author
**ImmyAfridi**

GitHub: https://github.com/ImmyAfridi
