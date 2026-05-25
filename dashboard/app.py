from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "outputs" / "features"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

REQUIRED_FILES = {
    "Module 1 feature sample": FEATURE_DIR / "module1_engineered_features_sample.csv",
    "Module 2 comparison": REPORT_DIR / "module2_imbalance_strategy_comparison.csv",
    "Module 3 comparison": REPORT_DIR / "module3_cost_sensitive_model_comparison.csv",
    "Module 3 threshold search": REPORT_DIR / "module3_threshold_search_results.csv",
    "Module 4 top-K anomaly comparison": REPORT_DIR / "module4_topk_anomaly_comparison.csv",
    "Module 4 supervised vs unsupervised": REPORT_DIR / "module4_supervised_vs_unsupervised_comparison.csv",
    "Module 5 explanation index": REPORT_DIR / "module5_transaction_explanations_index.csv",
    "Module 5 plain-English explanations": REPORT_DIR / "module5_plain_english_explanations.txt",
    "Module 5 SHAP importance": REPORT_DIR / "module5_shap_global_importance.csv",
    "Module 3 model metadata": MODEL_DIR / "module3_best_model_metadata.csv",
}

FIGURES = {
    "module2_metrics": FIGURE_DIR / "module2_model_metric_comparison.png",
    "module3_threshold_curve": FIGURE_DIR / "module3_cost_threshold_curve.png",
    "module3_default_vs_optimized": FIGURE_DIR / "module3_default_vs_optimized_cost.png",
    "module4_precision_at_k": FIGURE_DIR / "module4_precision_at_k_comparison.png",
    "module5_global_importance": FIGURE_DIR / "module5_shap_global_importance.png",
    "module5_summary": FIGURE_DIR / "module5_shap_summary_plot.png",
    "module5_waterfall": FIGURE_DIR / "module5_shap_waterfall_transaction.png",
}


def file_exists(path: Path) -> bool:
    """Return True when an artifact exists locally."""
    return path.is_file()


@st.cache_data(show_spinner=False)
def load_csv_safely(path: str) -> pd.DataFrame | None:
    """Load a CSV if present; otherwise return None so the app does not crash."""
    csv_path = Path(path)
    if not file_exists(csv_path):
        return None
    return pd.read_csv(csv_path)


@st.cache_data(show_spinner=False)
def load_text_safely(path: str) -> str | None:
    """Load a text artifact if present."""
    text_path = Path(path)
    if not file_exists(text_path):
        return None
    return text_path.read_text(encoding="utf-8")


def show_missing_artifact_warning() -> None:
    """Warn when expected dashboard artifacts are missing."""
    missing = [name for name, path in REQUIRED_FILES.items() if not file_exists(path)]
    if not missing:
        return

    st.warning("Dashboard artifacts are missing. Run Modules 1–5 first.")
    st.write("Missing artifacts:")
    st.write(", ".join(missing))
    st.code(
        "\n".join(
            [
                "python modules/module1_feature_engineering.py",
                "python modules/module2_imbalance_baseline.py",
                "python modules/module3_cost_sensitive.py",
                "python modules/module4_anomaly_detection.py",
                "python modules/module5_shap_explainability.py",
            ]
        ),
        language="bash",
    )


def format_value(value, value_type: str = "text") -> str:
    """Format KPI values for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Not available — run Module X."
    if value_type == "currency":
        return f"${value:,.0f}"
    if value_type == "percent":
        return f"{value:.2f}%"
    if value_type == "decimal":
        return f"{value:.4f}"
    return str(value)


def show_image_if_available(path: Path, caption: str) -> None:
    """Display a saved figure when available."""
    if file_exists(path):
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} is not available yet.")


def show_overview() -> None:
    """Render the portfolio overview view."""
    features = load_csv_safely(str(REQUIRED_FILES["Module 1 feature sample"]))
    module2 = load_csv_safely(str(REQUIRED_FILES["Module 2 comparison"]))
    module3 = load_csv_safely(str(REQUIRED_FILES["Module 3 comparison"]))
    module4_topk = load_csv_safely(str(REQUIRED_FILES["Module 4 top-K anomaly comparison"]))
    module4_comparison = load_csv_safely(
        str(REQUIRED_FILES["Module 4 supervised vs unsupervised"])
    )
    metadata = load_csv_safely(str(REQUIRED_FILES["Module 3 model metadata"]))

    st.header("Portfolio Overview")

    total_transactions = len(features) if features is not None else None
    fraud_rate = features["isFraud"].mean() * 100 if features is not None else None

    threshold = None
    if metadata is not None and "optimized_threshold" in metadata.columns:
        threshold = float(metadata.loc[0, "optimized_threshold"])

    default_cost = optimized_cost = cost_reduction = percent_reduction = None
    best_recall = best_precision = None
    if module3 is not None and "model_name" in module3.columns:
        optimized_rows = module3[
            module3["model_name"].astype(str).str.endswith("cost_optimized")
        ]
        if not optimized_rows.empty and "total_cost" in optimized_rows.columns:
            best_optimized = optimized_rows.sort_values("total_cost").iloc[0]
            base_model = str(best_optimized["model_name"]).replace("_cost_optimized", "")
            default_rows = module3[module3["model_name"] == f"{base_model}_default_0_5"]
            optimized_cost = float(best_optimized["total_cost"])
            best_recall = float(best_optimized.get("recall", np.nan))
            best_precision = float(best_optimized.get("precision", np.nan))
            if not default_rows.empty:
                default_cost = float(default_rows.iloc[0]["total_cost"])
                cost_reduction = default_cost - optimized_cost
                percent_reduction = (
                    cost_reduction / default_cost * 100 if default_cost else None
                )

    row1 = st.columns(4)
    row1[0].metric("Total Transactions", format_value(total_transactions))
    row1[1].metric("Fraud Rate", format_value(fraud_rate, "percent"))
    row1[2].metric("Best Threshold", format_value(threshold, "decimal"))
    row1[3].metric("Best Recall", format_value(best_recall, "decimal"))

    row2 = st.columns(4)
    row2[0].metric("Default Cost", format_value(default_cost, "currency"))
    row2[1].metric("Optimized Cost", format_value(optimized_cost, "currency"))
    row2[2].metric("Cost Reduction", format_value(cost_reduction, "currency"))
    row2[3].metric("Cost Reduction %", format_value(percent_reduction, "percent"))

    st.metric("Best Model Precision", format_value(best_precision, "decimal"))

    st.subheader("A. Accuracy Trap and Imbalance")
    if module2 is not None:
        st.dataframe(module2, use_container_width=True)
    else:
        st.info("Module 2 report not available — run Module 2.")
    show_image_if_available(FIGURES["module2_metrics"], "Module 2 model metric comparison")

    st.subheader("B. Cost-Sensitive Modeling")
    st.write(
        "The optimized threshold is chosen by minimizing expected business cost, "
        "not by maximizing accuracy."
    )
    if module3 is not None:
        st.dataframe(module3, use_container_width=True)
        cost_chart = module3[
            module3["model_name"].astype(str).str.contains("default|optimized", case=False)
        ]
        if {"model_name", "total_cost"}.issubset(cost_chart.columns):
            fig = px.bar(
                cost_chart,
                x="model_name",
                y="total_cost",
                title="Total Cost by Model Threshold Strategy",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Module 3 report not available — run Module 3.")
    show_image_if_available(FIGURES["module3_threshold_curve"], "Module 3 cost threshold curve")
    show_image_if_available(
        FIGURES["module3_default_vs_optimized"],
        "Module 3 default vs optimized cost",
    )

    st.subheader("C. Novel Fraud Monitoring")
    st.write(
        "Anomaly detection is used as a monitoring layer for new fraud patterns, "
        "not a replacement for the supervised model."
    )
    if module4_topk is not None:
        st.dataframe(module4_topk, use_container_width=True)
    else:
        st.info("Module 4 top-K report not available — run Module 4.")
    if module4_comparison is not None:
        st.dataframe(module4_comparison, use_container_width=True)
    show_image_if_available(
        FIGURES["module4_precision_at_k"],
        "Module 4 precision at K comparison",
    )

    st.subheader("D. Explainability")
    st.write("SHAP helps fraud analysts understand why the model flagged a transaction.")
    show_image_if_available(
        FIGURES["module5_global_importance"],
        "Module 5 SHAP global importance",
    )
    show_image_if_available(FIGURES["module5_summary"], "Module 5 SHAP summary plot")


def show_transaction_explainer() -> None:
    """Render the single transaction explainer view."""
    explanation_index = load_csv_safely(
        str(REQUIRED_FILES["Module 5 explanation index"])
    )
    explanation_text = load_text_safely(
        str(REQUIRED_FILES["Module 5 plain-English explanations"])
    )

    st.header("Single Transaction Explainer")

    if explanation_index is None:
        st.warning("Transaction explanations are missing. Run Module 5 first.")
        st.code("python modules/module5_shap_explainability.py", language="bash")
        show_image_if_available(
            FIGURES["module5_waterfall"],
            "Example local SHAP explanation",
        )
        return

    transaction_ids = explanation_index["TransactionID"].astype(str).tolist()
    selected_from_list = st.selectbox("Select TransactionID", transaction_ids)
    manual_transaction_id = st.text_input("Or type a TransactionID")
    selected_transaction_id = manual_transaction_id.strip() or selected_from_list

    selected_rows = explanation_index[
        explanation_index["TransactionID"].astype(str) == selected_transaction_id
    ]

    if selected_rows.empty:
        st.warning("TransactionID not found in the Module 5 explanation index.")
    else:
        row = selected_rows.iloc[0]
        recommendation = str(row["recommendation"]).lower()
        color = {"approve": "green", "review": "orange", "decline": "red"}.get(
            recommendation,
            "gray",
        )

        st.subheader(f"Transaction {row['TransactionID']}")
        cols = st.columns(5)
        cols[0].metric("Actual Label", int(row["y_true"]))
        cols[1].metric("Fraud Probability", f"{row['fraud_probability']:.2%}")
        cols[2].metric("Predicted Label", int(row["predicted_label"]))
        cols[3].markdown(
            f"<h3 style='color:{color}; margin-top: 0;'>{recommendation.title()}</h3>",
            unsafe_allow_html=True,
        )
        cols[4].caption("Recommendation")

    st.subheader("Example Local Explanation")
    show_image_if_available(
        FIGURES["module5_waterfall"],
        "Module 5 local SHAP explanation example",
    )

    st.subheader("Plain-English Sample Explanations")
    if explanation_text:
        st.text(explanation_text)
        if selected_rows.empty or selected_transaction_id not in explanation_text:
            st.info(
                "Detailed plain-English explanation is available for sample "
                "transactions generated in Module 5. This dashboard can be "
                "extended to generate text dynamically."
            )
    else:
        st.info("Plain-English explanations are not available — run Module 5.")

    st.subheader("Top 10 Riskiest Transactions")
    top_risky = explanation_index.sort_values("fraud_probability", ascending=False).head(10)
    st.dataframe(top_risky, use_container_width=True)


def show_methodology() -> None:
    """Render recruiter-friendly methodology notes and Q&A."""
    st.header("Project Methodology")

    st.subheader("Pipeline")
    st.write(
        "**Module 1: Feature engineering** turns transaction behavior into fraud-risk "
        "signals such as timing, velocity, amount deviation, device patterns, and missingness."
    )
    st.write(
        "**Module 2: Class imbalance** shows why high accuracy can still miss fraud "
        "and compares recall-focused baseline strategies."
    )
    st.write(
        "**Module 3: Cost-sensitive modeling** trains LightGBM and XGBoost models "
        "and chooses a threshold by minimizing expected business cost."
    )
    st.write(
        "**Module 4: Anomaly detection** monitors unusual behavior for novel fraud "
        "patterns that may not have labels yet."
    )
    st.write(
        "**Module 5: SHAP explanations** translates model behavior into global and "
        "transaction-level explanations for analysts."
    )
    st.write(
        "**Dashboard: Analyst workflow** brings metrics, business cost, anomaly "
        "monitoring, and transaction explanations into one review interface."
    )

    st.subheader("Interview Q&A")
    qa_items = [
        (
            "Why is accuracy bad for fraud detection?",
            "Fraud is rare, so a model can look accurate by predicting almost everything as non-fraud.",
        ),
        (
            "Why does AUC-PR matter?",
            "AUC-PR focuses on precision and recall for the minority fraud class.",
        ),
        (
            "Why does F2-score matter?",
            "F2 puts more weight on recall, which matches the high cost of missed fraud.",
        ),
        (
            "Why does threshold optimization matter?",
            "The best decision threshold depends on business costs, not just model probability output.",
        ),
        (
            "Why is anomaly detection useful?",
            "It can flag unusual behavior when fraud labels are delayed or new fraud patterns appear.",
        ),
        (
            "Why does SHAP matter?",
            "It helps analysts understand why a transaction was flagged and what evidence drove the score.",
        ),
    ]
    for question, answer in qa_items:
        st.markdown(f"**{question}**")
        st.write(answer)


def main() -> None:
    """Run the FraudLens Streamlit dashboard."""
    st.set_page_config(page_title="FraudLens Dashboard", layout="wide")
    st.title("FraudLens — Cost-Sensitive Fraud Detection Dashboard")
    st.write(
        "A fraud analyst dashboard for reviewing model performance, business cost, "
        "anomaly detection, and transaction-level explanations."
    )

    show_missing_artifact_warning()

    view = st.sidebar.radio(
        "Navigation",
        ["Portfolio Overview", "Single Transaction Explainer", "Project Methodology"],
    )

    if view == "Portfolio Overview":
        show_overview()
    elif view == "Single Transaction Explainer":
        show_transaction_explainer()
    else:
        show_methodology()


if __name__ == "__main__":
    main()
