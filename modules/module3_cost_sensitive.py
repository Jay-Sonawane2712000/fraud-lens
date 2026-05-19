from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features"
FIGURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "saved"

SAMPLE_FEATURE_PATH = FEATURE_OUTPUT_DIR / "module1_engineered_features_sample.csv"
FULL_FEATURE_PATH = FEATURE_OUTPUT_DIR / "module1_engineered_features.csv"

FALSE_NEGATIVE_COST = 500
FALSE_POSITIVE_COST = 10
TRUE_POSITIVE_VALUE = 500
TRUE_NEGATIVE_VALUE = 0
RANDOM_STATE = 42


def load_features() -> pd.DataFrame:
    """Load engineered Module 1 features, preferring the lightweight sample."""
    feature_path = SAMPLE_FEATURE_PATH if SAMPLE_FEATURE_PATH.is_file() else FULL_FEATURE_PATH
    if not feature_path.is_file():
        raise FileNotFoundError(
            "Missing engineered features. Run Module 1 before Module 3."
        )

    features = pd.read_csv(feature_path)
    print(f"Loaded feature file: {feature_path}")
    print(f"Loaded shape: {features.shape}")

    fraud_count = int(features["isFraud"].sum())
    fraud_rate = features["isFraud"].mean() * 100
    print(f"Fraud count: {fraud_count}")
    print(f"Fraud rate: {fraud_rate:.2f}%")

    return features


def prepare_model_data(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float, float]:
    """Prepare numeric Module 1 features for cost-sensitive modeling."""
    if "isFraud" not in features.columns:
        raise ValueError("Engineered feature matrix must contain isFraud.")

    y = features["isFraud"].astype(int)

    # This module uses Module 1 engineered features, not raw Kaggle columns.
    # TransactionID and TransactionDT remain useful for lookup and ordering, but
    # they are not model inputs for fraud-risk scoring.
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
    # Stratification matters with rare fraud labels because the train and test
    # splits both need a realistic fraud rate for cost evaluation.

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians).fillna(0)
    X_test = X_test.fillna(train_medians).fillna(0)

    positive_count = int(y_train.sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / positive_count
    fraud_rate = y.mean() * 100

    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Training fraud count: {positive_count}")
    print(f"Training non-fraud count: {negative_count}")
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    return X_train, X_test, y_train, y_test, scale_pos_weight, fraud_rate


def train_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float
) -> LGBMClassifier:
    """Train a simple cost-sensitive LightGBM classifier."""
    # scale_pos_weight tells the model to pay more attention to the minority
    # fraud class during training.
    model = LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float
) -> XGBClassifier:
    """Train a simple cost-sensitive XGBoost classifier."""
    # scale_pos_weight tells the model to pay more attention to the minority
    # fraud class during training.
    model = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def calculate_business_cost(
    y_true: pd.Series, y_prob: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Calculate the fraud business cost matrix for one decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    fn_cost = fn * FALSE_NEGATIVE_COST
    fp_cost = fp * FALSE_POSITIVE_COST
    total_cost = fn_cost + fp_cost
    total_value_saved = tp * TRUE_POSITIVE_VALUE + tn * TRUE_NEGATIVE_VALUE
    net_business_impact = total_value_saved - total_cost

    return {
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_cost": float(total_cost),
        "average_cost_per_transaction": float(total_cost / len(y_true)),
        "total_value_saved": float(total_value_saved),
        "net_business_impact": float(net_business_impact),
    }


def evaluate_threshold(
    model_name: str, y_true: pd.Series, y_prob: np.ndarray, threshold: float
) -> dict[str, float | int | str]:
    """Evaluate classification and business metrics at one threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    cost_metrics = calculate_business_cost(y_true, y_prob, threshold)

    # The default 0.5 threshold is rarely correct in fraud detection because
    # false negatives and false positives do not have equal cost.
    # Threshold optimization converts probability scores into business decisions.
    return {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "auc_pr": average_precision_score(y_true, y_prob),
        **cost_metrics,
    }


def search_thresholds(
    model_name: str, y_true: pd.Series, y_prob: np.ndarray
) -> tuple[pd.DataFrame, dict[str, float | int | str], dict[str, float | int | str]]:
    """Search thresholds from 0.01 to 0.99 and minimize total cost."""
    threshold_rows = [
        evaluate_threshold(model_name, y_true, y_prob, threshold)
        for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2)
    ]
    threshold_results = pd.DataFrame(threshold_rows)
    default_result = evaluate_threshold(f"{model_name}_default_0_5", y_true, y_prob, 0.5)

    best_threshold_row = (
        threshold_results.sort_values(["total_cost", "threshold"], ascending=[True, True])
        .iloc[0]
        .to_dict()
    )
    optimized_result = evaluate_threshold(
        f"{model_name}_cost_optimized",
        y_true,
        y_prob,
        float(best_threshold_row["threshold"]),
    )

    cost_difference = default_result["total_cost"] - optimized_result["total_cost"]
    print(f"\n{model_name} default 0.5 threshold:")
    print(
        f"  threshold={default_result['threshold']:.2f}, "
        f"total_cost=${default_result['total_cost']:,.0f}, "
        f"average_cost=${default_result['average_cost_per_transaction']:.2f}, "
        f"precision={default_result['precision']:.4f}, "
        f"recall={default_result['recall']:.4f}, "
        f"F2={default_result['f2']:.4f}, "
        f"TN={default_result['tn']}, FP={default_result['fp']}, "
        f"FN={default_result['fn']}, TP={default_result['tp']}"
    )
    print(f"{model_name} cost-optimized threshold:")
    print(
        f"  threshold={optimized_result['threshold']:.2f}, "
        f"total_cost=${optimized_result['total_cost']:,.0f}, "
        f"average_cost=${optimized_result['average_cost_per_transaction']:.2f}, "
        f"precision={optimized_result['precision']:.4f}, "
        f"recall={optimized_result['recall']:.4f}, "
        f"F2={optimized_result['f2']:.4f}, "
        f"TN={optimized_result['tn']}, FP={optimized_result['fp']}, "
        f"FN={optimized_result['fn']}, TP={optimized_result['tp']}"
    )
    print(f"  Dollar difference in total_cost: ${cost_difference:,.0f}")

    threshold_results["model_name"] = model_name
    return threshold_results, default_result, optimized_result


def create_visualizations(
    threshold_results: pd.DataFrame,
    comparison_results: pd.DataFrame,
    best_model_name: str,
) -> None:
    """Create Module 3 cost-sensitive modeling figures."""
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for model_name, model_rows in threshold_results.groupby("model_name"):
        plt.plot(
            model_rows["threshold"],
            model_rows["total_cost"],
            label=f"{model_name} total cost",
        )
        best_row = model_rows.sort_values("total_cost").iloc[0]
        plt.scatter(
            best_row["threshold"],
            best_row["total_cost"],
            s=70,
            marker="o",
            label=f"{model_name} best threshold={best_row['threshold']:.2f}",
        )
    plt.title("Expected Fraud Cost by Classification Threshold")
    plt.xlabel("Fraud probability threshold")
    plt.ylabel("Total expected cost ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module3_cost_threshold_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    cost_rows = comparison_results[
        comparison_results["model_name"].isin(
            [
                "lightgbm_default_0_5",
                "lightgbm_cost_optimized",
                "xgboost_default_0_5",
                "xgboost_cost_optimized",
            ]
        )
    ]
    plt.bar(cost_rows["model_name"], cost_rows["total_cost"], color="#5c7c99")
    plt.title("Default vs Cost-Optimized Threshold Cost")
    plt.xlabel("Model threshold strategy")
    plt.ylabel("Total expected cost ($)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module3_default_vs_optimized_cost.png", dpi=150)
    plt.close()

    best_threshold_rows = threshold_results[
        threshold_results["model_name"] == best_model_name
    ]
    plt.figure(figsize=(10, 5))
    plt.plot(
        best_threshold_rows["threshold"],
        best_threshold_rows["precision"],
        label="Precision",
    )
    plt.plot(
        best_threshold_rows["threshold"],
        best_threshold_rows["recall"],
        label="Recall",
    )
    plt.title(f"Precision-Recall Tradeoff by Threshold: {best_model_name}")
    plt.xlabel("Fraud probability threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module3_precision_recall_tradeoff.png", dpi=150)
    plt.close()


def save_reports(threshold_results: pd.DataFrame, comparison_results: pd.DataFrame) -> None:
    """Save threshold-search and model-comparison reports."""
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    threshold_report_path = REPORT_OUTPUT_DIR / "module3_threshold_search_results.csv"
    comparison_report_path = REPORT_OUTPUT_DIR / "module3_cost_sensitive_model_comparison.csv"

    threshold_results.to_csv(threshold_report_path, index=False)
    comparison_results.to_csv(comparison_report_path, index=False)

    print(f"\nThreshold search report saved to: {threshold_report_path}")
    print(f"Model comparison report saved to: {comparison_report_path}")
    print(f"Figures saved to: {FIGURE_OUTPUT_DIR}")


def save_best_model(
    best_model,
    best_result: pd.Series,
    fraud_rate: float,
    scale_pos_weight: float,
) -> None:
    """Save the best cost-sensitive model and metadata."""
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_OUTPUT_DIR / "module3_best_cost_sensitive_model.pkl"
    metadata_path = MODEL_OUTPUT_DIR / "module3_best_model_metadata.csv"

    joblib.dump(best_model, model_path)

    metadata = pd.DataFrame(
        [
            {
                "model_name": best_result["model_name"],
                "optimized_threshold": best_result["threshold"],
                "total_cost": best_result["total_cost"],
                "average_cost_per_transaction": best_result[
                    "average_cost_per_transaction"
                ],
                "total_value_saved": best_result["total_value_saved"],
                "net_business_impact": best_result["net_business_impact"],
                "auc_pr": best_result["auc_pr"],
                "f2": best_result["f2"],
                "fraud_rate": fraud_rate,
                "scale_pos_weight": scale_pos_weight,
            }
        ]
    )
    metadata.to_csv(metadata_path, index=False)

    print(f"Best model saved to: {model_path}")
    print(f"Best model metadata saved to: {metadata_path}")


def main() -> None:
    features = load_features()
    X_train, X_test, y_train, y_test, scale_pos_weight, fraud_rate = prepare_model_data(
        features
    )

    print("\nTraining LightGBM...")
    lightgbm_model = train_lightgbm(X_train, y_train, scale_pos_weight)
    lightgbm_prob = lightgbm_model.predict_proba(X_test)[:, 1]

    print("Training XGBoost...")
    xgboost_model = train_xgboost(X_train, y_train, scale_pos_weight)
    xgboost_prob = xgboost_model.predict_proba(X_test)[:, 1]

    (
        lightgbm_thresholds,
        lightgbm_default,
        lightgbm_optimized,
    ) = search_thresholds("lightgbm", y_test, lightgbm_prob)
    (
        xgboost_thresholds,
        xgboost_default,
        xgboost_optimized,
    ) = search_thresholds("xgboost", y_test, xgboost_prob)

    threshold_results = pd.concat(
        [lightgbm_thresholds, xgboost_thresholds], ignore_index=True
    )
    comparison_results = pd.DataFrame(
        [lightgbm_default, lightgbm_optimized, xgboost_default, xgboost_optimized]
    )

    best_optimized_result = comparison_results[
        comparison_results["model_name"].str.endswith("cost_optimized")
    ].sort_values("total_cost", ascending=True).iloc[0]

    best_base_model_name = str(best_optimized_result["model_name"]).replace(
        "_cost_optimized", ""
    )
    best_model = lightgbm_model if best_base_model_name == "lightgbm" else xgboost_model
    default_model_name = f"{best_base_model_name}_default_0_5"
    best_default_result = comparison_results[
        comparison_results["model_name"] == default_model_name
    ].iloc[0]

    create_visualizations(threshold_results, comparison_results, best_base_model_name)
    save_reports(threshold_results, comparison_results)
    save_best_model(best_model, best_optimized_result, fraud_rate, scale_pos_weight)

    cost_reduction = (
        best_default_result["total_cost"] - best_optimized_result["total_cost"]
    )
    percent_reduction = (
        cost_reduction / best_default_result["total_cost"] * 100
        if best_default_result["total_cost"] > 0
        else 0
    )

    print("\nModule 3 Business Summary")
    print(f"Best model name: {best_optimized_result['model_name']}")
    print(f"Best optimized threshold: {best_optimized_result['threshold']:.2f}")
    print(
        f"Cost at default 0.5 threshold: "
        f"${best_default_result['total_cost']:,.0f}"
    )
    print(
        f"Cost at optimized threshold: "
        f"${best_optimized_result['total_cost']:,.0f}"
    )
    print(f"Dollar reduction in expected loss: ${cost_reduction:,.0f}")
    print(f"Percent reduction in expected loss: {percent_reduction:.2f}%")
    print(f"Best model recall: {best_optimized_result['recall']:.4f}")
    print(f"Best model precision: {best_optimized_result['precision']:.4f}")
    print(
        "Threshold optimization matters in fraud detection because the decision "
        "boundary should reflect the much higher cost of missed fraud compared "
        "with review friction."
    )
    print(
        f"Cost-sensitive {best_base_model_name} reduced expected fraud-review "
        f"loss by ${cost_reduction:,.0f} ({percent_reduction:.2f}%) vs default "
        "threshold on the test sample."
    )


if __name__ == "__main__":
    main()
