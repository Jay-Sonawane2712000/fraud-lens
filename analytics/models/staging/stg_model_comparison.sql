select *
from read_csv_auto('{{ var("source_root") }}/outputs/reports/module3_cost_sensitive_model_comparison.csv', header = true)
