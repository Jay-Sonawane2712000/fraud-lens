from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import TomekLinks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features"
FIGURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"

SAMPLE_FEATURE_PATH = FEATURE_OUTPUT_DIR / "module1_engineered_features_sample.csv"
FULL_FEATURE_PATH = FEATURE_OUTPUT_DIR / "module1_engineered_features.csv"

RANDOM_STATE = 42


def load_features() -> pd.DataFrame:
    """Load engineered Module 1 features, preferring the lightweight sample."""
    feature_path = SAMPLE_FEATURE_PATH if SAMPLE_FEATURE_PATH.is_file() else FULL_FEATURE_PATH
    if not feature_path.is_file():
        raise FileNotFoundError(
            "Missing engineered features. Run Module 1 before Module 2."
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare a simple numeric baseline modeling matrix."""
    if "isFraud" not in features.columns:
        raise ValueError("Engineered feature matrix must contain isFraud.")

    y = features["isFraud"].astype(int)

    # TransactionID and TransactionDT are useful lookup/order fields, but they
    # are not fraud-risk features for this baseline model input.
    lookup_columns = ["TransactionID", "TransactionDT"]
    X = features.drop(columns=lookup_columns + ["isFraud"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    # This is a modeling baseline, not final production preprocessing. Median
    # imputation keeps the comparison focused on imbalance strategies.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    # Stratification matters because fraud is rare; each split needs a similar
    # fraud rate so recall and F2 comparisons are meaningful.

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians).fillna(0)
    X_test = X_test.fillna(train_medians).fillna(0)

    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Training fraud rate: {y_train.mean() * 100:.2f}%")
    print(f"Test fraud rate: {y_test.mean() * 100:.2f}%")

    return X_train, X_test, y_train, y_test


def evaluate_predictions(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | int | str]:
    """Calculate fraud-relevant evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # F2 weights recall higher than precision, which fits fraud detection
    # because missed fraud is usually more expensive than a false alarm.
    # AUC-PR is more informative for imbalanced fraud problems because it
    # focuses on performance for the minority fraud class.
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "auc_pr": average_precision_score(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print(f"\n{model_name}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  F2: {metrics['f2']:.4f}")
    print(f"  AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"  Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return metrics


def logistic_regression(class_weight: str | None = None) -> LogisticRegression:
    """Create a consistent logistic regression model."""
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def run_naive_baseline(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict[str, float | int | str]:
    """Train logistic regression without imbalance handling."""
    print("\nNaive Analyst Baseline: Accuracy Trap")
    print(
        "Accuracy can look high because non-fraud dominates the dataset. "
        "Fraud recall is the key warning signal because missed fraud is costly."
    )

    # Logistic regression is sensitive to feature scale, so this baseline uses
    # standard scaling before fitting the model.
    pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("model", logistic_regression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    return evaluate_predictions("naive", y_test, y_pred, y_score)


def run_class_weight_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict[str, float | int | str]:
    """Train logistic regression with balanced class weights."""
    pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            ("model", logistic_regression(class_weight="balanced")),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    return evaluate_predictions("class_weight_balanced", y_test, y_pred, y_score)


def run_smote_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict[str, float | int | str]:
    """Train logistic regression after SMOTE oversampling."""
    pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            # Resampling is applied only to training data. Resampling test data
            # would leak synthetic or edited label information into evaluation.
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", logistic_regression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    return evaluate_predictions("smote", y_test, y_pred, y_score)


def run_tomek_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> dict[str, float | int | str]:
    """Train logistic regression after Tomek Links undersampling."""
    pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            # Tomek Links cleans only the training set. Changing the test set
            # would make the test distribution unrealistically easy.
            ("tomek", TomekLinks()),
            ("model", logistic_regression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_score = pipeline.predict_proba(X_test)[:, 1]
    return evaluate_predictions("tomek_links", y_test, y_pred, y_score)


def run_cross_validation(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Run stratified CV with F2 scoring for corrected imbalance strategies."""
    print("\nCross-validation F2 scores on training data:")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

    pipelines = {
        "class_weight_balanced": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("model", logistic_regression(class_weight="balanced")),
            ]
        ),
        "smote_logistic_regression": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                # SMOTE must happen inside each CV fold so validation rows are
                # never used to create synthetic training examples.
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", logistic_regression()),
            ]
        ),
        "tomek_logistic_regression": ImbPipeline(
            [
                ("scaler", StandardScaler()),
                # Tomek undersampling also belongs inside each fold to avoid
                # using validation-set structure during training cleanup.
                ("tomek", TomekLinks()),
                ("model", logistic_regression()),
            ]
        ),
    }

    cv_rows = []
    for model_name, pipeline in pipelines.items():
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring=f2_scorer,
            cv=cv,
            n_jobs=None,
        )
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"  {model_name}: mean={mean_score:.4f}, std={std_score:.4f}")
        cv_rows.append(
            {
                "model_name": model_name,
                "mean_f2": mean_score,
                "std_f2": std_score,
            }
        )

    return pd.DataFrame(cv_rows)


def create_visualizations(results: pd.DataFrame, cv_results: pd.DataFrame) -> None:
    """Save Module 2 metric comparison figures."""
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_results = results.set_index("model_name")
    metric_columns = ["precision", "recall", "f2", "auc_pr"]

    x = np.arange(len(plot_results.index))
    width = 0.2
    plt.figure(figsize=(12, 6))
    for idx, metric_name in enumerate(metric_columns):
        plt.bar(
            x + (idx - 1.5) * width,
            plot_results[metric_name],
            width=width,
            label=metric_name,
        )
    plt.title("Fraud Model Metric Comparison")
    plt.xlabel("Approach")
    plt.ylabel("Score")
    plt.xticks(x, plot_results.index, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module2_model_metric_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    x = np.arange(len(plot_results.index))
    plt.bar(x - 0.18, plot_results["fn"], width=0.36, label="Missed fraud (FN)")
    plt.bar(x + 0.18, plot_results["fp"], width=0.36, label="False alarms (FP)")
    plt.title("Missed Fraud vs False Alarm Tradeoff")
    plt.xlabel("Approach")
    plt.ylabel("Count")
    plt.xticks(x, plot_results.index, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURE_OUTPUT_DIR / "module2_confusion_matrix_comparison.png", dpi=150
    )
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(
        cv_results["model_name"],
        cv_results["mean_f2"],
        yerr=cv_results["std_f2"],
        capsize=5,
        color="#5c7c99",
    )
    plt.title("Cross-validated F2 Scores")
    plt.xlabel("Corrected imbalance strategy")
    plt.ylabel("Mean F2 score")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module2_cv_f2_scores.png", dpi=150)
    plt.close()


def save_reports(results: pd.DataFrame, cv_results: pd.DataFrame) -> None:
    """Save Module 2 comparison reports."""
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    strategy_report_path = REPORT_OUTPUT_DIR / "module2_imbalance_strategy_comparison.csv"
    cv_report_path = REPORT_OUTPUT_DIR / "module2_cv_f2_scores.csv"

    results.to_csv(strategy_report_path, index=False)
    cv_results.to_csv(cv_report_path, index=False)

    print(f"\nStrategy report saved to: {strategy_report_path}")
    print(f"CV report saved to: {cv_report_path}")
    print(f"Figures saved to: {FIGURE_OUTPUT_DIR}")


def main() -> None:
    features = load_features()
    X_train, X_test, y_train, y_test = prepare_model_data(features)

    result_rows = [
        run_naive_baseline(X_train, X_test, y_train, y_test),
        run_class_weight_model(X_train, X_test, y_train, y_test),
        run_smote_model(X_train, X_test, y_train, y_test),
        run_tomek_model(X_train, X_test, y_train, y_test),
    ]

    results = pd.DataFrame(result_rows)
    cv_results = run_cross_validation(X_train, y_train)

    create_visualizations(results, cv_results)
    save_reports(results, cv_results)

    best_f2_model = results.sort_values("f2", ascending=False).iloc[0]
    highest_recall_model = results.sort_values("recall", ascending=False).iloc[0]

    print("\nModule 2 Business Summary")
    print(
        f"Best test F2-score: {best_f2_model['model_name']} "
        f"({best_f2_model['f2']:.4f})"
    )
    print(
        f"Highest test recall: {highest_recall_model['model_name']} "
        f"({highest_recall_model['recall']:.4f})"
    )
    print(
        "Naive accuracy is misleading because the model can be mostly right by "
        "predicting the dominant non-fraud class while still missing fraud."
    )
    print(
        "F2 is selected instead of accuracy because it emphasizes recall, and "
        "missed fraud is usually more expensive than an extra review alert."
    )
    print(
        "AUC-PR is used instead of AUC-ROC because it focuses on precision and "
        "recall for the rare fraud class."
    )
    print(
        "This module shows that evaluating fraud models by accuracy can hide "
        "missed fraud risk."
    )


if __name__ == "__main__":
    main()
