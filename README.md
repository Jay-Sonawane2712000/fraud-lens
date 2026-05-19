FraudLens helps financial teams reduce fraud losses by prioritizing suspicious transactions where mistakes are most expensive.

# FraudLens — Cost-Sensitive Fraud Detection and Anomaly Explanation System

## Business Scenario

Fraud teams must catch high-risk transactions without overwhelming reviewers or blocking too many legitimate customers. This project will build a cost-sensitive fraud detection workflow that compares baseline imbalance handling, business-cost-aware modeling, anomaly detection, and model explanations for operational decision support.

## Planned Modules

1. **Feature Engineering**: Prepare transaction-level features and reusable preprocessing outputs.
2. **Imbalance Baseline**: Train baseline fraud classifiers with class imbalance handling.
3. **Cost-Sensitive Modeling**: Optimize models around business costs such as false negatives and false positives.
4. **Anomaly Detection**: Identify unusual transaction behavior that may not be captured by supervised labels.
5. **SHAP Explainability**: Explain model predictions and surface fraud drivers for stakeholder review.

## Dataset Note

The preferred dataset is the IEEE-CIS Fraud Detection dataset from Kaggle. PaySim is a fallback only if Kaggle access is unavailable.

Raw data must not be committed to GitHub. Keep downloaded datasets in `data/raw/` locally.

## Temporary Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
