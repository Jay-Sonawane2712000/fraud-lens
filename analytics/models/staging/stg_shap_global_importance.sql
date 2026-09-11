select *
from read_csv_auto('{{ var("source_root") }}/outputs/reports/module5_shap_global_importance.csv', header = true)
