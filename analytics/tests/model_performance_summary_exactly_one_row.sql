select count(*) as row_count
from {{ ref('model_performance_summary') }}
having count(*) != 1
