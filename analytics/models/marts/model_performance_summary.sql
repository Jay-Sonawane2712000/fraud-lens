with metadata as (
    select *
    from {{ ref('stg_best_model_metadata') }}
),

optimized as (
    select c.*
    from {{ ref('stg_model_comparison') }} as c
    inner join metadata as m
        on c.model_name = m.model_name
),

default_threshold as (
    select
        c.total_cost as default_total_cost,
        c.threshold as default_threshold
    from {{ ref('stg_model_comparison') }} as c
    inner join metadata as m
        on c.model_name = replace(m.model_name, '_cost_optimized', '_default_0_5')
)

select
    m.model_name,
    m.optimized_threshold,
    o.recall,
    o.precision,
    coalesce(o.f2, m.f2) as f2,
    coalesce(o.auc_pr, m.auc_pr) as auc_pr,
    coalesce(o.total_cost, m.total_cost) as total_cost,
    coalesce(o.total_value_saved, m.total_value_saved) as total_value_saved,
    coalesce(o.net_business_impact, m.net_business_impact) as net_business_impact,
    m.fraud_rate,
    case
        when d.default_total_cost is null then 'default threshold comparison unavailable'
        when coalesce(o.total_cost, m.total_cost) < d.default_total_cost
            then 'optimized threshold reduced expected cost vs default'
        when coalesce(o.total_cost, m.total_cost) = d.default_total_cost
            then 'optimized threshold matched default expected cost'
        else 'optimized threshold increased expected cost vs default'
    end as comparison_label_vs_default_threshold
from metadata as m
left join optimized as o
    on m.model_name = o.model_name
left join default_threshold as d
    on true
