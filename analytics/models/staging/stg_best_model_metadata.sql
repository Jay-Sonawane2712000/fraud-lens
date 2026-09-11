select *
from read_csv_auto('{{ var("source_root") }}/models/saved/module3_best_model_metadata.csv', header = true)
