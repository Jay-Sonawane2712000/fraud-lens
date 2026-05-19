from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features"
FIGURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"

SAMPLE_FEATURE_PATH = FEATURE_OUTPUT_DIR / "module1_engineered_features_sample.csv"
FULL_FEATURE_PATH = FEATURE_OUTPUT_DIR / "module1_engineered_features.csv"
MODULE3_REPORT_PATH = REPORT_OUTPUT_DIR / "module3_cost_sensitive_model_comparison.csv"

RANDOM_STATE = 42
TOP_K_VALUES = [100, 250, 500, 1000]


def load_features() -> pd.DataFrame:
    """Load engineered Module 1 features, preferring the lightweight sample."""
    feature_path = SAMPLE_FEATURE_PATH if SAMPLE_FEATURE_PATH.is_file() else FULL_FEATURE_PATH
    if not feature_path.is_file():
        raise FileNotFoundError(
            "Missing engineered features. Run Module 1 before Module 4."
        )

    features = pd.read_csv(feature_path)
    print(f"Loaded feature file: {feature_path}")
    print(f"Loaded shape: {features.shape}")

    fraud_count = int(features["isFraud"].sum())
    fraud_rate = features["isFraud"].mean() * 100
    print(f"Fraud count: {fraud_count}")
    print(f"Fraud rate: {fraud_rate:.2f}%")

    return features


def prepare_data(
    features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, float]:
    """Prepare scaled data for unsupervised anomaly detection."""
    if "isFraud" not in features.columns:
        raise ValueError("Engineered feature matrix must contain isFraud.")

    # Labels are used only after scoring to evaluate whether anomaly detection
    # surfaces fraud. They are not used to train the unsupervised models.
    y = features["isFraud"].astype(int)

    # Module 4 uses Module 1 engineered features, not raw Kaggle columns.
    # TransactionID and TransactionDT are lookup/order fields, not anomaly inputs.
    X = features.drop(columns=["TransactionID", "TransactionDT", "isFraud"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    # Stratification keeps the rare fraud rate stable for evaluation.

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians).fillna(0)
    X_test = X_test.fillna(train_medians).fillna(0)

    # Distance and reconstruction-based methods are sensitive to feature scale,
    # so the scaler is fit on training data only and then applied to test data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # In real fraud operations, an anomaly detector is often fit on mostly-normal
    # historical behavior to detect transactions that deviate from normal.
    normal_train_mask = y_train.to_numpy() == 0
    X_train_normal_scaled = X_train_scaled[normal_train_mask]

    training_fraud_rate = y_train.mean()
    contamination = max(training_fraud_rate, 0.03)

    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Training fraud rate: {training_fraud_rate * 100:.2f}%")
    print(f"Test fraud rate: {y_test.mean() * 100:.2f}%")
    print(f"Non-fraud records used for unsupervised fitting: {len(X_train_normal_scaled)}")
    print(f"Anomaly contamination setting: {contamination:.4f}")

    return X_train_normal_scaled, X_test_scaled, y_train, y_test, contamination


def train_isolation_forest(
    X_train_normal_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    contamination: float,
) -> np.ndarray:
    """Train Isolation Forest and return higher-is-more-suspicious scores."""
    # Isolation Forest is useful for high-dimensional tabular anomaly detection
    # because unusual points are easier to isolate.
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_normal_scaled)
    return -model.decision_function(X_test_scaled)


def train_lof(
    X_train_normal_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    contamination: float,
) -> np.ndarray:
    """Train Local Outlier Factor and return higher-is-more-suspicious scores."""
    # LOF captures local density deviations, which can detect transactions
    # unusual relative to nearby normal behavior.
    model = LocalOutlierFactor(
        n_neighbors=35,
        novelty=True,
        contamination=contamination,
    )
    model.fit(X_train_normal_scaled)
    return -model.decision_function(X_test_scaled)


def train_autoencoder(
    X_train_normal_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
) -> np.ndarray:
    """Train a minimal Keras autoencoder and return reconstruction errors."""
    keras.utils.set_random_seed(RANDOM_STATE)
    input_dim = X_train_normal_scaled.shape[1]

    # Autoencoders learn to reconstruct normal behavior; high reconstruction
    # error suggests an unusual transaction.
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(input_dim, activation="linear"),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train_normal_scaled,
        X_train_normal_scaled,
        epochs=20,
        batch_size=256,
        validation_split=0.1,
        verbose=0,
    )

    reconstructed = model.predict(X_test_scaled, verbose=0)
    return np.mean((X_test_scaled - reconstructed) ** 2, axis=1)


def evaluate_top_k(
    method_name: str, y_test: pd.Series, anomaly_scores: np.ndarray
) -> pd.DataFrame:
    """Evaluate fraud capture among the highest-scored anomalies."""
    # Top-K evaluation matches analyst workflow because fraud teams can manually
    # review only a limited number of highest-risk transactions.
    total_fraud = int(y_test.sum())
    test_fraud_rate = y_test.mean()
    sorted_indices = np.argsort(anomaly_scores)[::-1]

    rows = []
    for k in TOP_K_VALUES:
        actual_k = min(k, len(y_test))
        top_indices = sorted_indices[:actual_k]
        fraud_found = int(y_test.iloc[top_indices].sum())
        precision_at_k = fraud_found / actual_k
        recall_at_k = fraud_found / total_fraud if total_fraud else 0
        fraud_lift = precision_at_k / test_fraud_rate if test_fraud_rate else 0
        rows.append(
            {
                "method": method_name,
                "k": k,
                "flagged_count": actual_k,
                "fraud_found": fraud_found,
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "fraud_lift": fraud_lift,
            }
        )

    return pd.DataFrame(rows)


def summarize_scores(
    method_name: str, y_test: pd.Series, anomaly_scores: np.ndarray
) -> dict[str, float | str]:
    """Summarize anomaly score distributions for fraud and non-fraud."""
    fraud_scores = anomaly_scores[y_test.to_numpy() == 1]
    non_fraud_scores = anomaly_scores[y_test.to_numpy() == 0]
    return {
        "method": method_name,
        "mean_anomaly_score_fraud": float(np.mean(fraud_scores)),
        "mean_anomaly_score_non_fraud": float(np.mean(non_fraud_scores)),
        "median_anomaly_score_fraud": float(np.median(fraud_scores)),
        "median_anomaly_score_non_fraud": float(np.median(non_fraud_scores)),
    }


def build_supervised_vs_unsupervised_comparison(
    topk_results: pd.DataFrame,
) -> pd.DataFrame:
    """Build final comparison table across monitoring approaches."""
    comparison_rows = []
    precision_at_500 = topk_results[topk_results["k"] == 500]

    business_use_cases = {
        "isolation_forest": "High-dimensional tabular anomaly monitoring for unusual transactions.",
        "local_outlier_factor": "Local density monitoring for transactions unusual relative to nearby normal behavior.",
        "autoencoder": "Reconstruction-error monitoring for transactions that do not resemble normal behavior.",
    }

    for _, row in precision_at_500.iterrows():
        comparison_rows.append(
            {
                "method": row["method"],
                "type": "unsupervised",
                "main_metric": "precision_at_500",
                "main_metric_value": row["precision_at_k"],
                "business_use_case": business_use_cases.get(row["method"], "Anomaly monitoring."),
            }
        )

    if MODULE3_REPORT_PATH.is_file():
        supervised_results = pd.read_csv(MODULE3_REPORT_PATH)
        cost_optimized = supervised_results[
            supervised_results["model_name"].astype(str).str.endswith("cost_optimized")
        ]
        if not cost_optimized.empty and "total_cost" in cost_optimized.columns:
            best_supervised = cost_optimized.sort_values("total_cost").iloc[0]
            comparison_rows.append(
                {
                    "method": "supervised_cost_sensitive_best",
                    "type": "supervised",
                    "main_metric": "optimized_total_cost",
                    "main_metric_value": best_supervised["total_cost"],
                    "business_use_case": "Known-fraud classification with explicit business cost optimization.",
                }
            )
        elif "f2" in supervised_results.columns:
            best_supervised = supervised_results.sort_values("f2", ascending=False).iloc[0]
            comparison_rows.append(
                {
                    "method": "supervised_cost_sensitive_best",
                    "type": "supervised",
                    "main_metric": "f2",
                    "main_metric_value": best_supervised["f2"],
                    "business_use_case": "Known-fraud classification using recall-weighted model selection.",
                }
            )
    else:
        print(
            "\nModule 3 comparison report not found; skipping supervised comparison row."
        )

    return pd.DataFrame(comparison_rows)


def create_visualizations(
    y_test: pd.Series,
    score_by_method: dict[str, np.ndarray],
    topk_results: pd.DataFrame,
    best_unsupervised_method: str,
) -> None:
    """Save Module 4 anomaly detection figures."""
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_scores = score_by_method[best_unsupervised_method]
    fraud_scores = best_scores[y_test.to_numpy() == 1]
    non_fraud_scores = best_scores[y_test.to_numpy() == 0]

    plt.figure(figsize=(10, 5))
    plt.hist(non_fraud_scores, bins=50, alpha=0.6, label="Non-fraud", color="#4c78a8")
    plt.hist(fraud_scores, bins=50, alpha=0.6, label="Fraud", color="#d65f5f")
    plt.title(f"Anomaly Score Distribution: {best_unsupervised_method}")
    plt.xlabel("Anomaly score (higher means more suspicious)")
    plt.ylabel("Transaction count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module4_anomaly_score_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    for method_name, method_rows in topk_results.groupby("method"):
        plt.plot(
            method_rows["k"],
            method_rows["precision_at_k"],
            marker="o",
            label=method_name,
        )
    plt.title("Precision at K for Anomaly Detection Methods")
    plt.xlabel("Top K transactions reviewed")
    plt.ylabel("Precision at K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module4_precision_at_k_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    for method_name, method_rows in topk_results.groupby("method"):
        plt.plot(
            method_rows["k"],
            method_rows["recall_at_k"],
            marker="o",
            label=method_name,
        )
    plt.title("Recall at K for Anomaly Detection Methods")
    plt.xlabel("Top K transactions reviewed")
    plt.ylabel("Recall at K")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module4_recall_at_k_comparison.png", dpi=150)
    plt.close()


def save_reports(
    topk_results: pd.DataFrame,
    score_summary: pd.DataFrame,
    final_comparison: pd.DataFrame,
) -> None:
    """Save Module 4 reports."""
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    topk_path = REPORT_OUTPUT_DIR / "module4_topk_anomaly_comparison.csv"
    score_summary_path = REPORT_OUTPUT_DIR / "module4_anomaly_score_summary.csv"
    comparison_path = REPORT_OUTPUT_DIR / "module4_supervised_vs_unsupervised_comparison.csv"

    topk_results.to_csv(topk_path, index=False)
    score_summary.to_csv(score_summary_path, index=False)
    final_comparison.to_csv(comparison_path, index=False)

    print(f"\nTop-K anomaly report saved to: {topk_path}")
    print(f"Anomaly score summary saved to: {score_summary_path}")
    print(f"Supervised vs unsupervised comparison saved to: {comparison_path}")
    print(f"Figures saved to: {FIGURE_OUTPUT_DIR}")


def main() -> None:
    features = load_features()
    (
        X_train_normal_scaled,
        X_test_scaled,
        _,
        y_test,
        contamination,
    ) = prepare_data(features)

    print("\nTraining Isolation Forest...")
    isolation_scores = train_isolation_forest(
        X_train_normal_scaled, X_test_scaled, contamination
    )

    print("Training Local Outlier Factor...")
    lof_scores = train_lof(X_train_normal_scaled, X_test_scaled, contamination)

    print("Training Keras autoencoder...")
    autoencoder_scores = train_autoencoder(X_train_normal_scaled, X_test_scaled)

    score_by_method = {
        "isolation_forest": isolation_scores,
        "local_outlier_factor": lof_scores,
        "autoencoder": autoencoder_scores,
    }

    topk_results = pd.concat(
        [
            evaluate_top_k(method_name, y_test, anomaly_scores)
            for method_name, anomaly_scores in score_by_method.items()
        ],
        ignore_index=True,
    )
    score_summary = pd.DataFrame(
        [
            summarize_scores(method_name, y_test, anomaly_scores)
            for method_name, anomaly_scores in score_by_method.items()
        ]
    )

    print("\nAnomaly score distribution summary:")
    for _, row in score_summary.iterrows():
        print(
            f"  {row['method']}: "
            f"mean fraud={row['mean_anomaly_score_fraud']:.4f}, "
            f"mean non-fraud={row['mean_anomaly_score_non_fraud']:.4f}, "
            f"median fraud={row['median_anomaly_score_fraud']:.4f}, "
            f"median non-fraud={row['median_anomaly_score_non_fraud']:.4f}"
        )

    print("\nTop-K fraud capture:")
    for _, row in topk_results.iterrows():
        print(
            f"  {row['method']} K={int(row['k'])}: "
            f"fraud_found={int(row['fraud_found'])}, "
            f"precision={row['precision_at_k']:.4f}, "
            f"recall={row['recall_at_k']:.4f}, "
            f"lift={row['fraud_lift']:.2f}x"
        )

    final_comparison = build_supervised_vs_unsupervised_comparison(topk_results)

    best_row = (
        topk_results[topk_results["k"] == 500]
        .sort_values(["precision_at_k", "recall_at_k"], ascending=[False, False])
        .iloc[0]
    )
    best_unsupervised_method = best_row["method"]

    create_visualizations(y_test, score_by_method, topk_results, best_unsupervised_method)
    save_reports(topk_results, score_summary, final_comparison)

    print("\nModule 4 Business Summary")
    print(f"Best unsupervised method based on precision_at_500: {best_unsupervised_method}")
    print(f"precision_at_500: {best_row['precision_at_k']:.4f}")
    print(f"recall_at_500: {best_row['recall_at_k']:.4f}")
    print(f"fraud_lift_at_500: {best_row['fraud_lift']:.2f}x")
    print(
        "Anomaly detection is useful even when supervised models perform better "
        "because it can monitor for behavior that does not look like historical "
        "normal activity."
    )
    print(
        "The cold start problem in fraud means new fraud patterns may not have "
        "labeled historical examples yet, so supervised models may not recognize "
        "them immediately."
    )
    print(
        "I used unsupervised anomaly detection as a monitoring layer for novel "
        "fraud patterns, not as a replacement for supervised classification."
    )


if __name__ == "__main__":
    main()
