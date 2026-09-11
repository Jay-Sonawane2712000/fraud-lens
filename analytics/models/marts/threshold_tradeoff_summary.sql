with best_model as (
    select
        replace(model_name, '_cost_optimized', '') as threshold_model_name
    from {{ ref('stg_best_model_metadata') }}
),

thresholds as (
    select
        t.model_name,
        t.threshold,
        t.recall,
        t.precision,
        t.f2,
        t.auc_pr,
        t.total_cost,
        t.fp + t.tp as review_volume,
        t.total_value_saved,
        t.net_business_impact
    from {{ ref('stg_threshold_search') }} as t
    inner join best_model as b
        on t.model_name = b.threshold_model_name
)

select *
from thresholds
