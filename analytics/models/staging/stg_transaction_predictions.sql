select *
from read_csv_auto('{{ var("source_root") }}/outputs/reports/module5_transaction_explanations_index.csv', header = true)
