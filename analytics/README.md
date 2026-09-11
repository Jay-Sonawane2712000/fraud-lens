# FraudLens Analytics

This folder contains the lightweight dbt + DuckDB analytics layer for FraudLens.

The current step builds only staging tables from existing Module 1-5 artifacts. Final analytics marts will be added later.

## Windows / VS Code Commands

Run these commands from the project root:

```powershell
cd analytics
dbt debug --profiles-dir .
dbt run --profiles-dir .
dbt test --profiles-dir .
```

DuckDB tables are written to:

```text
analytics/fraud_lens.duckdb
```

## Current Staging Sources

- `../outputs/features/module1_engineered_features.csv`
- `../outputs/reports/module5_transaction_explanations_index.csv`
- `../outputs/reports/module3_threshold_search_results.csv`
- `../models/saved/module3_best_model_metadata.csv`
- `../outputs/reports/module3_cost_sensitive_model_comparison.csv`
- `../outputs/reports/module4_topk_anomaly_comparison.csv`
- `../outputs/reports/module4_supervised_vs_unsupervised_comparison.csv`
- `../outputs/reports/module5_shap_global_importance.csv`

`min_segment_size` is configured as `100` in `dbt_project.yml` for later mart work.
