from pathlib import Path
import pandas as pd

def main():
    data_path = Path(__file__).parent / "creditcard.csv"
    df = pd.read_csv(data_path)

    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- First Five Rows ---")
    print(df.head())

    print("\n--- Column Information ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum().sum())

    print("\n--- Class Distribution ---")
    print(df["Class"].value_counts())

    print("\nClass Percentages:")
    print((df["Class"].value_counts(normalize=True) * 100).round(4))

    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())

    print("\n--- Transaction Amount Summary ---")
    print(df["Amount"].describe())




if __name__ == "__main__":
    main()