from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features"
FIGURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "saved"

SAMPLE_FEATURE_PATH = FEATURE_OUTPUT_DIR / "module1_engineered_features_sample.csv"
FULL_FEATURE_PATH = FEATURE_OUTPUT_DIR / "module1_engineered_features.csv"
MODEL_PATH = MODEL_OUTPUT_DIR / "module3_best_cost_sensitive_model.pkl"
METADATA_PATH = MODEL_OUTPUT_DIR / "module3_best_model_metadata.csv"

RANDOM_STATE = 42


def load_features() -> pd.DataFrame:
    """Load engineered Module 1 features for explanation."""
    feature_path = SAMPLE_FEATURE_PATH if SAMPLE_FEATURE_PATH.is_file() else FULL_FEATURE_PATH
    if not feature_path.is_file():
        raise FileNotFoundError(
            "Missing engineered features. Run Module 1 before Module 5."
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, float]:
    """Prepare model-ready data while preserving TransactionID for explanations."""
    if "isFraud" not in features.columns:
        raise ValueError("Engineered feature matrix must contain isFraud.")

    transaction_ids = features["TransactionID"].copy()
    y = features["isFraud"].astype(int)

    # Module 5 explains engineered fraud-risk features from Module 1, not raw
    # Kaggle columns. TransactionID is retained only for analyst lookup.
    X = features.drop(columns=["TransactionID", "TransactionDT", "isFraud"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    X_train, X_test, y_train, y_test, _, test_transaction_ids = train_test_split(
        X,
        y,
        transaction_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    # Stratification matters because fraud is rare; the test set needs the same
    # class balance for credible explanation examples and evaluation.

    train_medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(train_medians).fillna(0)
    X_test = X_test.fillna(train_medians).fillna(0)

    fraud_rate = y.mean() * 100
    print(f"Training shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Test fraud rate: {y_test.mean() * 100:.2f}%")

    return X_train, X_test, y_train, y_test, test_transaction_ids.reset_index(drop=True), fraud_rate


def load_or_train_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[LGBMClassifier, str]:
    """Load Module 3 model when available, otherwise train a simple LightGBM."""
    if MODEL_PATH.is_file():
        model = joblib.load(MODEL_PATH)
        print(f"Loaded Module 3 best model from: {MODEL_PATH}")
        return model, "loaded from Module 3 artifact"

    positive_count = int(y_train.sum())
    negative_count = int((y_train == 0).sum())
    scale_pos_weight = negative_count / positive_count

    print("Module 3 model artifact missing; retraining simple LightGBM fallback.")
    print(f"Fallback scale_pos_weight: {scale_pos_weight:.2f}")
    model = LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model, "retrained LightGBM fallback"


def load_threshold() -> float:
    """Load Module 3 optimized threshold, falling back to 0.5."""
    if METADATA_PATH.is_file():
        metadata = pd.read_csv(METADATA_PATH)
        if "optimized_threshold" in metadata.columns and not metadata.empty:
            threshold = float(metadata.loc[0, "optimized_threshold"])
            print(f"Loaded optimized threshold from metadata: {threshold:.2f}")
            return threshold

    print("Warning: Module 3 threshold metadata missing; using threshold = 0.50.")
    return 0.5


def compute_shap_values(
    model, X_sample: pd.DataFrame
) -> tuple[np.ndarray, float, shap.TreeExplainer]:
    """Compute SHAP values for a manageable explanation sample."""
    # SHAP estimates how much each feature contributes to pushing a prediction
    # higher toward fraud risk or lower toward normal behavior.
    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_sample)

    if isinstance(raw_shap_values, list):
        shap_values = raw_shap_values[1] if len(raw_shap_values) > 1 else raw_shap_values[0]
    elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
        shap_values = raw_shap_values[:, :, 1]
    else:
        shap_values = raw_shap_values

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_array = np.asarray(expected_value).ravel()
        expected_value = float(expected_array[1] if len(expected_array) > 1 else expected_array[0])
    else:
        expected_value = float(expected_value)

    return np.asarray(shap_values), expected_value, explainer


def create_global_importance_plot(
    shap_values: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """Save mean absolute SHAP global feature importance."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    top_features = importance.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["mean_abs_shap"], color="#5c7c99")
    plt.title("Top Global SHAP Feature Importance")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module5_shap_global_importance.png", dpi=150)
    plt.close()

    return importance


def create_summary_plot(shap_values: np.ndarray, X_sample: pd.DataFrame) -> None:
    """Save SHAP summary plot for the top 15 features."""
    top_indices = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:15]
    X_top = X_sample.iloc[:, top_indices]
    shap_top = shap_values[:, top_indices]

    plt.figure()
    shap.summary_plot(shap_top, X_top, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_DIR / "module5_shap_summary_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_waterfall_plot(
    shap_values: np.ndarray,
    expected_value: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    predicted_labels: np.ndarray,
    transaction_ids: pd.Series,
    threshold: float,
) -> tuple[int, int]:
    """Save local explanation for one flagged transaction."""
    flagged_fraud_mask = (y_test.reset_index(drop=True) == 1) & (predicted_labels == 1)
    if flagged_fraud_mask.any():
        selected_position = int(np.where(flagged_fraud_mask.to_numpy())[0][0])
    else:
        selected_position = int(np.argmax(probabilities))

    transaction_id = int(transaction_ids.iloc[selected_position])
    shap_row = shap_values[selected_position]
    feature_values = X_test.reset_index(drop=True).iloc[selected_position]

    try:
        explanation = shap.Explanation(
            values=shap_row,
            base_values=expected_value,
            data=feature_values.to_numpy(),
            feature_names=X_test.columns.tolist(),
        )
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.tight_layout()
        plt.savefig(
            FIGURE_OUTPUT_DIR / "module5_shap_waterfall_transaction.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Waterfall plot created for TransactionID: {transaction_id}")
    except Exception as exc:
        print(f"Waterfall plot failed; using fallback local bar chart. Reason: {exc}")
        top_indices = np.argsort(np.abs(shap_row))[::-1][:10]
        top_indices = top_indices[::-1]
        plt.figure(figsize=(10, 6))
        colors = ["#d65f5f" if shap_row[i] > 0 else "#4c78a8" for i in top_indices]
        plt.barh(X_test.columns[top_indices], shap_row[top_indices], color=colors)
        plt.title(f"Local SHAP Contributions for Transaction {transaction_id}")
        plt.xlabel("SHAP contribution")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(FIGURE_OUTPUT_DIR / "module5_shap_waterfall_transaction.png", dpi=150)
        plt.close()

    return selected_position, transaction_id


def create_dependence_plots(
    shap_values: np.ndarray, X_sample: pd.DataFrame, top_features: list[str]
) -> None:
    """Save dependence plots for the top three global SHAP features."""
    for plot_number, feature_name in enumerate(top_features[:3], start=1):
        output_path = FIGURE_OUTPUT_DIR / f"module5_shap_dependence_{plot_number}.png"
        try:
            shap.dependence_plot(
                feature_name,
                shap_values,
                X_sample,
                show=False,
                interaction_index=None,
            )
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            print(
                f"Dependence plot failed for {feature_name}; using fallback scatter. "
                f"Reason: {exc}"
            )
            feature_index = X_sample.columns.get_loc(feature_name)
            plt.figure(figsize=(8, 5))
            plt.scatter(
                X_sample[feature_name],
                shap_values[:, feature_index],
                alpha=0.4,
                s=12,
            )
            plt.title(f"SHAP Dependence: {feature_name}")
            plt.xlabel(feature_name)
            plt.ylabel("SHAP value")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()


def generate_plain_english_explanation(
    transaction_row: pd.Series,
    shap_row: np.ndarray,
    feature_names: list[str],
    probability: float,
    threshold: float,
    transaction_id: int,
) -> str:
    """Create a non-technical analyst explanation for one transaction."""
    positive_indices = np.where(shap_row > 0)[0]
    if len(positive_indices) == 0:
        top_indices = np.argsort(np.abs(shap_row))[::-1][:3]
    else:
        top_indices = positive_indices[np.argsort(shap_row[positive_indices])[::-1]][:3]

    if probability >= threshold + 0.20:
        action = "decline"
    elif probability >= threshold:
        action = "review"
    else:
        action = "approve"

    lines = [f"Transaction #{transaction_id} was flagged because:"]
    for index in top_indices:
        feature_name = feature_names[index]
        feature_value = transaction_row[feature_name]
        lines.append(
            f"- {feature_name} had value {feature_value:.4f} and increased fraud risk."
        )
    lines.append(f"Combined fraud probability: {probability * 100:.2f}%.")
    lines.append(f"Recommended action: {action}.")
    return "\n".join(lines)


def save_reports(
    explanation_index: pd.DataFrame,
    importance: pd.DataFrame,
    explanations: list[str],
) -> None:
    """Save Module 5 explanation reports."""
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    explanation_index_path = REPORT_OUTPUT_DIR / "module5_transaction_explanations_index.csv"
    importance_path = REPORT_OUTPUT_DIR / "module5_shap_global_importance.csv"
    explanations_path = REPORT_OUTPUT_DIR / "module5_plain_english_explanations.txt"

    explanation_index.to_csv(explanation_index_path, index=False)
    importance.to_csv(importance_path, index=False)
    explanations_path.write_text("\n\n".join(explanations), encoding="utf-8")

    print(f"Transaction explanation index saved to: {explanation_index_path}")
    print(f"Global SHAP importance saved to: {importance_path}")
    print(f"Plain-English explanations saved to: {explanations_path}")


def main() -> None:
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    features = load_features()
    X_train, X_test, y_train, y_test, transaction_ids, _ = prepare_model_data(features)
    model, model_source = load_or_train_model(X_train, y_train)
    threshold = load_threshold()

    probabilities = model.predict_proba(X_test)[:, 1]
    predicted_labels = (probabilities >= threshold).astype(int)
    recommendations = np.where(
        probabilities >= threshold + 0.20,
        "decline",
        np.where(probabilities >= threshold, "review", "approve"),
    )

    explanation_index = pd.DataFrame(
        {
            "TransactionID": transaction_ids,
            "y_true": y_test.reset_index(drop=True),
            "fraud_probability": probabilities,
            "predicted_label": predicted_labels,
            "recommendation": recommendations,
        }
    )

    shap_sample_size = min(1000, len(X_test))
    X_shap_sample = X_test.head(shap_sample_size).copy()
    shap_values, expected_value, _ = compute_shap_values(model, X_shap_sample)

    importance = create_global_importance_plot(shap_values, X_shap_sample.columns.tolist())
    create_summary_plot(shap_values, X_shap_sample)

    sample_probabilities = probabilities[:shap_sample_size]
    sample_predictions = predicted_labels[:shap_sample_size]
    sample_y_test = y_test.head(shap_sample_size)
    sample_transaction_ids = transaction_ids.head(shap_sample_size)

    selected_position, selected_transaction_id = create_waterfall_plot(
        shap_values,
        expected_value,
        X_shap_sample,
        sample_y_test,
        sample_probabilities,
        sample_predictions,
        sample_transaction_ids,
        threshold,
    )

    top_features = importance.head(5)["feature"].tolist()
    create_dependence_plots(shap_values, X_shap_sample, top_features[:3])

    explanation_positions = [selected_position]
    high_probability_positions = list(np.argsort(sample_probabilities)[::-1])
    for position in high_probability_positions:
        if position not in explanation_positions:
            explanation_positions.append(int(position))
        if len(explanation_positions) == 3:
            break

    explanations = [
        generate_plain_english_explanation(
            X_shap_sample.reset_index(drop=True).iloc[position],
            shap_values[position],
            X_shap_sample.columns.tolist(),
            float(sample_probabilities[position]),
            threshold,
            int(sample_transaction_ids.iloc[position]),
        )
        for position in explanation_positions
    ]

    save_reports(explanation_index, importance, explanations)

    print("\nModule 5 Business Summary")
    print(f"Model source: {model_source}")
    print(f"Threshold used: {threshold:.2f}")
    print(f"Number of test transactions explained in index: {len(explanation_index)}")
    print(f"SHAP global/local computation rows: {shap_sample_size}")
    print(f"Selected transaction ID for local explanation: {selected_transaction_id}")
    print(f"Top 5 global SHAP features: {', '.join(top_features)}")
    print(f"Figures saved to: {FIGURE_OUTPUT_DIR}")
    print(f"Reports saved to: {REPORT_OUTPUT_DIR}")
    print(
        "SHAP is more useful than generic feature importance for fraud analysts "
        "because it explains both the overall drivers of model behavior and the "
        "specific reasons an individual transaction was pushed toward fraud risk "
        "or normal behavior."
    )


if __name__ == "__main__":
    main()
