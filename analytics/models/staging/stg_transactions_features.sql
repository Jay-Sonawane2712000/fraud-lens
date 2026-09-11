select *
from read_csv_auto('{{ var("source_root") }}/outputs/features/module1_engineered_features.csv', header = true)
