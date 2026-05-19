from pathlib import Path

import pandas as pd


RAW_DATA_DIR = Path("data/raw")

REQUIRED_RAW_FILES = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv",
    "sample_submission.csv",
]


def print_missing_value_summary(frame: pd.DataFrame, label: str) -> None:
    missing_pct = frame.isna().mean().mul(100).sort_values(ascending=False).head(10)
    print(f"\nTop 10 missing-value percentages in {label}:")
    for column_name, pct in missing_pct.items():
        print(f"  {column_name}: {pct:.2f}%")


def main() -> None:
    print(f"Project path: {Path.cwd()}")
    print(f"Raw data path: {(Path.cwd() / RAW_DATA_DIR).resolve()}")

    missing_files = [
        file_name for file_name in REQUIRED_RAW_FILES if not (RAW_DATA_DIR / file_name).is_file()
    ]
    if missing_files:
        for file_name in missing_files:
            print(f"Missing required file: {RAW_DATA_DIR / file_name}")
        raise SystemExit(1)

    # train_transaction is the main supervised training table because it contains
    # transaction records, the TransactionID join key, and the isFraud target.
    train_transaction = pd.read_csv(RAW_DATA_DIR / "train_transaction.csv")

    # train_identity adds optional device, browser, email, and identity signals
    # that can help explain suspicious transaction behavior later in the project.
    train_identity = pd.read_csv(RAW_DATA_DIR / "train_identity.csv")

    print(f"\ntrain_transaction shape: {train_transaction.shape}")
    print(f"train_identity shape: {train_identity.shape}")
    print(f"train_transaction columns: {train_transaction.shape[1]}")
    print(f"train_identity columns: {train_identity.shape[1]}")

    transaction_id_in_transaction = "TransactionID" in train_transaction.columns
    transaction_id_in_identity = "TransactionID" in train_identity.columns
    print(f"TransactionID exists in train_transaction: {transaction_id_in_transaction}")
    print(f"TransactionID exists in train_identity: {transaction_id_in_identity}")

    if "isFraud" not in train_transaction.columns:
        print("Error: isFraud column is missing from train_transaction.csv.")
        raise SystemExit(1)

    fraud_count = int(train_transaction["isFraud"].sum())
    fraud_rate = train_transaction["isFraud"].mean() * 100

    # Fraud rate matters because fraud detection is a class imbalance problem:
    # fraudulent transactions are usually rare compared with legitimate ones.
    print(f"Fraud count: {fraud_count}")
    print(f"Fraud rate: {fraud_rate:.2f}%")

    print_missing_value_summary(train_transaction, "train_transaction")
    print_missing_value_summary(train_identity, "train_identity")

    print("\nSuccess: IEEE-CIS raw data loads correctly.")


if __name__ == "__main__":
    main()
