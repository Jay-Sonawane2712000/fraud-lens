with counts as (
    select
        (select count(*) from {{ ref('review_queue_summary') }}) as review_queue_count,
        (select count(*) from {{ ref('stg_transaction_predictions') }}) as prediction_count
)

select *
from counts
where review_queue_count <= 0
   or review_queue_count >= prediction_count
