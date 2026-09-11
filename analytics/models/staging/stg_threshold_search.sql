select *
from read_csv_auto('{{ var("source_root") }}/outputs/reports/module3_threshold_search_results.csv', header = true)
