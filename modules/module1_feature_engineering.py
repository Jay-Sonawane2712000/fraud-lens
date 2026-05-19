from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEATURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features"
FIGURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw IEEE-CIS training tables."""
    transaction_path = RAW_DATA_DIR / "train_transaction.csv"
    identity_path = RAW_DATA_DIR / "train_identity.csv"

    if not transaction_path.is_file():
        raise FileNotFoundError(f"Missing required file: {transaction_path}")
    if not identity_path.is_file():
        raise FileNotFoundError(f"Missing required file: {identity_path}")

    # train_transaction is the main supervised table because it contains the
    # isFraud label used to learn which transactions were confirmed fraud.
    train_transaction = pd.read_csv(transaction_path)

    # train_identity adds device, browser, email, and identity signals that may
    # reveal unusual login or checkout behavior around a transaction.
    train_identity = pd.read_csv(identity_path)

    print(f"train_transaction shape: {train_transaction.shape}")
    print(f"train_identity shape: {train_identity.shape}")

    fraud_count = int(train_transaction["isFraud"].sum())
    fraud_rate = train_transaction["isFraud"].mean() * 100
    print(f"Fraud count: {fraud_count}")
    print(f"Fraud rate: {fraud_rate:.2f}%")

    return train_transaction, train_identity


def add_velocity_features(features: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Create exact card-level transaction velocity features."""
    velocity_base = merged[["card1", "TransactionDT"]].copy()
    velocity_base["_row_position"] = np.arange(len(velocity_base))
    velocity_base = velocity_base.sort_values(["card1", "TransactionDT"], kind="mergesort")

    counts_1hr = np.zeros(len(velocity_base), dtype=np.int32)
    counts_6hr = np.zeros(len(velocity_base), dtype=np.int32)
    counts_24hr = np.zeros(len(velocity_base), dtype=np.int32)

    # Multiple transactions in a short time window can indicate account takeover,
    # stolen card testing, or bot-driven fraud attempts.
    for _, group in velocity_base.groupby("card1", dropna=False, sort=False):
        row_positions = group["_row_position"].to_numpy()
        times = group["TransactionDT"].to_numpy()

        for window_seconds, target_array in [
            (60 * 60, counts_1hr),
            (6 * 60 * 60, counts_6hr),
            (24 * 60 * 60, counts_24hr),
        ]:
            start_positions = np.searchsorted(
                times, times - window_seconds, side="left"
            )
            window_counts = np.arange(len(times)) - start_positions + 1
            target_array[row_positions] = window_counts

    features["card1_txn_count_total"] = (
        merged.groupby("card1", dropna=False)["TransactionID"].transform("count").astype(np.int32)
    )
    features["card1_txn_count_past_1hr"] = counts_1hr
    features["card1_txn_count_past_6hr"] = counts_6hr
    features["card1_txn_count_past_24hr"] = counts_24hr

    return features


def build_features(
    train_transaction: pd.DataFrame, train_identity: pd.DataFrame
) -> pd.DataFrame:
    """Build business-reasoned fraud detection features."""
    if "isFraud" not in train_transaction.columns:
        raise ValueError("train_transaction.csv must contain the isFraud target.")

    transaction_columns = train_transaction.columns.tolist()
    identity_columns = [
        column for column in train_identity.columns.tolist() if column != "TransactionID"
    ]

    # A left join keeps every labeled transaction, even when identity information
    # is missing for that transaction.
    merged = train_transaction.merge(
        train_identity, on="TransactionID", how="left", suffixes=("", "_identity")
    )

    features = merged[
        ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt"]
    ].copy()

    # Fraud may cluster at unusual hours. Week/day timing also helps capture
    # bursty, campaign-like fraud behavior over short windows.
    features["transaction_hour"] = (features["TransactionDT"] // 3600) % 24
    features["transaction_day"] = features["TransactionDT"] // (24 * 3600)
    features["transaction_week"] = features["transaction_day"] // 7

    # Fraud often appears as unusually large transactions or transactions that
    # are unusual for a cardholder. card1 is used as a cardholder proxy here.
    features["transaction_amt_log"] = np.log1p(features["TransactionAmt"])
    features["card1_avg_amt"] = merged.groupby("card1", dropna=False)[
        "TransactionAmt"
    ].transform("mean")
    features["card1_amt_deviation"] = (
        features["TransactionAmt"] - features["card1_avg_amt"]
    )

    safe_avg_amount = features["card1_avg_amt"].replace(0, np.nan)
    # Ratio-to-average is more interpretable than raw amount alone because it
    # shows how unusual a purchase is for the same cardholder proxy.
    features["card1_amt_ratio_to_avg"] = (
        features["TransactionAmt"] / safe_avg_amount
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    features = add_velocity_features(features, merged)

    # Missing or new device information may indicate risk. Mismatched purchaser
    # and recipient email domains can also signal suspicious behavior.
    features["has_identity_info"] = merged[identity_columns].notna().any(axis=1).astype(int)
    features["device_type_missing"] = merged["DeviceType"].isna().astype(int)
    features["device_info_missing"] = merged["DeviceInfo"].isna().astype(int)
    features["email_domain_match"] = (
        merged["P_emaildomain"].notna()
        & merged["R_emaildomain"].notna()
        & (merged["P_emaildomain"] == merged["R_emaildomain"])
    ).astype(int)

    email_combo = (
        merged["card1"].astype("string").fillna("missing_card")
        + "|"
        + merged["P_emaildomain"].astype("string").fillna("missing_email")
        + "|"
        + merged["R_emaildomain"].astype("string").fillna("missing_email")
    )
    device_combo = (
        merged["card1"].astype("string").fillna("missing_card")
        + "|"
        + merged["DeviceType"].astype("string").fillna("missing_device_type")
        + "|"
        + merged["DeviceInfo"].astype("string").fillna("missing_device_info")
    )

    # Rare card-email and card-device combinations can indicate a new device,
    # unusual checkout setup, or possible account takeover.
    features["card_email_combo_seen_count"] = email_combo.map(email_combo.value_counts())
    features["card_device_combo_seen_count"] = device_combo.map(device_combo.value_counts())

    # Missingness itself can be predictive in fraud datasets because fraudsters
    # may avoid providing stable identity signals.
    features["transaction_missing_count"] = merged[transaction_columns].isna().sum(axis=1)
    if identity_columns:
        features["identity_missing_count"] = merged[identity_columns].isna().sum(axis=1)

    return features


def save_correlation_report(features: pd.DataFrame) -> pd.DataFrame:
    """Save a preview of engineered feature correlations with fraud."""
    # Identifiers and raw timestamps are useful for joining, lookup, and
    # ordering, but they should not be interpreted as fraud-risk drivers.
    excluded_columns = ["TransactionID", "TransactionDT", "isFraud"]
    numeric_features = features.select_dtypes(include=[np.number]).drop(
        columns=excluded_columns, errors="ignore"
    )
    correlations = (
        numeric_features.corrwith(features["isFraud"])
        .dropna()
    )

    correlation_report = (
        pd.DataFrame(
            {
                "feature": correlations.index,
                "correlation_with_isFraud": correlations.values,
                "absolute_correlation": np.abs(correlations.values),
            }
        )
        .sort_values("absolute_correlation", ascending=False)
        .reset_index(drop=True)
    )

    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_OUTPUT_DIR / "module1_feature_correlation_preview.csv"
    correlation_report.to_csv(report_path, index=False)

    print("\nTop 15 absolute correlations with isFraud:")
    for _, row in correlation_report.head(15).iterrows():
        print(
            f"  {row['feature']}: "
            f"{row['correlation_with_isFraud']:.4f} "
            f"(abs={row['absolute_correlation']:.4f})"
        )

    return correlation_report


def create_visualizations(features: pd.DataFrame, correlation_report: pd.DataFrame) -> None:
    """Create descriptive Module 1 figures."""
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fraud_rate_by_hour = features.groupby("transaction_hour")["isFraud"].mean() * 100
    plt.figure(figsize=(10, 5))
    plt.bar(fraud_rate_by_hour.index, fraud_rate_by_hour.values, color="#2f6f9f")
    plt.title("Fraud Rate by Transaction Hour")
    plt.xlabel("Transaction hour from dataset clock")
    plt.ylabel("Fraud rate (%)")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module1_fraud_rate_by_hour.png", dpi=150)
    plt.close()

    top_correlations = correlation_report.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(
        top_correlations["feature"],
        top_correlations["absolute_correlation"],
        color="#6b8f3a",
    )
    plt.title("Top Engineered Feature Correlations with Fraud")
    plt.xlabel("Absolute correlation with isFraud")
    plt.ylabel("Engineered feature")
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module1_top_feature_correlations.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    non_fraud_amounts = features.loc[
        features["isFraud"] == 0, "transaction_amt_log"
    ].dropna()
    fraud_amounts = features.loc[
        features["isFraud"] == 1, "transaction_amt_log"
    ].dropna()
    plt.hist(non_fraud_amounts, bins=50, alpha=0.6, label="Non-fraud", color="#4c78a8")
    plt.hist(fraud_amounts, bins=50, alpha=0.6, label="Fraud", color="#d65f5f")
    plt.title("Log Transaction Amount Distribution by Fraud Label")
    plt.xlabel("log1p(TransactionAmt)")
    plt.ylabel("Transaction count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURE_OUTPUT_DIR / "module1_transaction_amount_distribution.png", dpi=150
    )
    plt.close()


def main() -> None:
    FEATURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_transaction, train_identity = load_data()
    features = build_features(train_transaction, train_identity)
    correlation_report = save_correlation_report(features)
    create_visualizations(features, correlation_report)

    feature_path = FEATURE_OUTPUT_DIR / "module1_engineered_features.csv"
    sample_path = FEATURE_OUTPUT_DIR / "module1_engineered_features_sample.csv"
    features.to_csv(feature_path, index=False)
    features.head(50_000).to_csv(sample_path, index=False)

    fraud_rate = features["isFraud"].mean() * 100
    top_5_features = correlation_report.head(5)["feature"].tolist()

    print("\nModule 1 Business Summary")
    print(f"Engineered feature matrix rows: {features.shape[0]}")
    print(f"Engineered feature matrix columns: {features.shape[1]}")
    print(f"Fraud rate: {fraud_rate:.2f}%")
    print(f"Top 5 strongest correlation features: {', '.join(top_5_features)}")
    print(
        "Naive raw-data modeling would miss behavior patterns such as unusual "
        "purchase size, rapid repeat transactions, missing identity signals, "
        "and rare card-device or card-email combinations."
    )
    print(
        "The engineered features capture timing, amount deviation, transaction "
        "velocity, identity coverage, device risk, email consistency, and "
        "missingness from a fraud-risk perspective."
    )
    print(f"Feature matrix saved to: {feature_path}")
    print(f"Feature sample saved to: {sample_path}")
    print(f"Correlation report saved to: {REPORT_OUTPUT_DIR / 'module1_feature_correlation_preview.csv'}")
    print(f"Figures saved to: {FIGURE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
