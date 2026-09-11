# FraudLens Portfolio Summary

## 30-Second Recruiter Summary

FraudLens is an end-to-end fraud analytics project built on the IEEE-CIS fraud transaction dataset. It tackles class imbalance, cost-sensitive fraud detection, anomaly monitoring, model explainability, and analyst decision support in one workflow. The project moves beyond accuracy by optimizing fraud decisions around business cost, then turns model outputs into dbt/DuckDB analytics marts and a Streamlit dashboard for review queues, threshold tradeoffs, and high-risk segment monitoring.

## Resume Bullet Options

- Built FraudLens, a cost-sensitive fraud detection and analytics system using Python, scikit-learn, LightGBM, XGBoost, SHAP, Streamlit, dbt, and DuckDB on the IEEE-CIS fraud dataset.
- Reduced expected fraud-review loss by approximately $8,490, or 16.79%, by optimizing the decision threshold to 0.24, achieving recall of 0.8413 versus a naive baseline recall of 0.0000.
- Created dbt/DuckDB fraud analytics marts and a Streamlit dashboard for model KPIs, threshold tradeoffs, review queues, high-risk segments, and explainability-driven analyst support.

## Interview Explanation - STAR

**Situation:** Fraud detection is highly imbalanced, and a model can appear accurate while missing costly fraudulent transactions.

**Task:** Build a portfolio-ready fraud system that detects fraud risk, handles class imbalance, optimizes decisions around business cost, explains predictions, and packages outputs for analyst review.

**Action:** I engineered transaction, timing, amount-deviation, velocity, identity, device, and missingness features; compared imbalance-aware baselines; trained cost-sensitive LightGBM and XGBoost models; optimized the classification threshold using a fraud cost matrix; added unsupervised anomaly detection for novel fraud monitoring; generated SHAP explanations; and built dbt/DuckDB marts plus a Streamlit dashboard.

**Result:** The best cost-sensitive workflow used an optimized threshold around 0.24, achieved recall around 0.8413 and precision around 0.0997, and reduced expected fraud-review loss by about $8,490, or 16.79%, versus the default threshold on the test sample.

## Technical Architecture Summary

- **Module 1 - Feature Engineering:** Built fraud-risk features from transaction timing, amount behavior, velocity, identity/device information, email-domain consistency, and missingness.
- **Module 2 - Imbalance Baselines:** Demonstrated the accuracy trap with logistic regression baselines and imbalance strategies such as class weighting, SMOTE, and Tomek Links.
- **Module 3 - Cost-Sensitive Optimization:** Trained LightGBM and XGBoost with fraud class weighting and optimized thresholds using explicit false-negative and false-positive costs.
- **Module 4 - Anomaly Detection:** Added Isolation Forest, Local Outlier Factor, and a minimal autoencoder for monitoring unusual behavior that may not match historical fraud labels.
- **Module 5 - SHAP Explainability:** Generated global and local SHAP artifacts for explaining fraud model behavior. The dashboard review queue uses `top_shap_driver_proxy`, which is based on global SHAP importance, not per-transaction SHAP contributions.
- **Module 6 - Analytics Marts:** Added a dbt + DuckDB layer that stages existing artifacts and builds fraud analytics marts for decision support.
- **Dashboard:** Built a Streamlit interface for portfolio overview, transaction explanations, analytics marts, and project methodology.

## Business Impact Summary

FraudLens frames fraud detection as a decision problem, not just a classification problem. The optimized model threshold prioritizes missed-fraud cost over raw accuracy, which improved recall and reduced expected business loss on the test sample. The dashboard and marts translate model outputs into analyst-friendly views: review queues, threshold tradeoffs, high-risk segment rankings, fraud-rate segment monitoring, and explainability summaries.

## Module 6 Analytics Validation

The dbt/DuckDB analytics layer was verified after implementation:

| Mart | Rows |
|---|---:|
| `fraud_rate_by_segment` | 78 |
| `threshold_tradeoff_summary` | 99 |
| `review_queue_summary` | 2,287 |
| `model_performance_summary` | 1 |
| `high_risk_segments` | 55 |

Validation results:

- `dbt run` built 13 models.
- `dbt test` passed 32 tests.
- The Streamlit dashboard successfully queries all five Module 6 marts.

## What I Would Improve Next

- Add transaction-level SHAP contribution tables so each review queue row can show transaction-specific drivers.
- Add real-time scoring and a lightweight API for batch or streaming transactions.
- Add model and data drift monitoring for changing fraud behavior.
- Expand anomaly monitoring with graph-based fraud features across cards, devices, emails, and identities.
- Deploy the dashboard and analytics database in a cloud environment with scheduled refreshes.

## Honest Limitations

- The project is local and portfolio-oriented; it is not a deployed production fraud system.
- The IEEE-CIS dataset is historical and anonymized, so some operational fields are unavailable.
- The review queue's `top_shap_driver_proxy` is a global SHAP feature proxy, not a transaction-specific SHAP explanation.
- Cost assumptions are simplified and fixed for demonstration; real fraud operations would tune costs with finance, risk, and operations teams.
- The dashboard reads generated artifacts and marts; it does not perform live model scoring.
