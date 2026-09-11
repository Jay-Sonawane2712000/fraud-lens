select *
from read_csv_auto('{{ var("source_root") }}/outputs/reports/module4_supervised_vs_unsupervised_comparison.csv', header = true)
