with threshold_config as (
    select optimized_threshold
    from {{ ref('stg_best_model_metadata') }}
),

top_shap_driver as (
    select feature as top_shap_driver_proxy
    from {{ ref('stg_shap_global_importance') }}
    order by mean_abs_shap desc
    limit 1
),

joined as (
    select
        p.TransactionID,
        p.fraud_probability,
        p.predicted_label,
        p.y_true,
        t.optimized_threshold,
        p.recommendation,
        case
            when p.fraud_probability >= 0.80 then 'review immediately'
            when p.fraud_probability >= 0.50 then 'review this shift'
            else 'monitor / queued review'
        end as recommended_action_label,
        s.top_shap_driver_proxy,
        f.TransactionAmt,
        f.transaction_hour,
        f.has_identity_info,
        f.email_domain_match
    from {{ ref('stg_transaction_predictions') }} as p
    inner join {{ ref('stg_transactions_features') }} as f
        on p.TransactionID = f.TransactionID
    cross join threshold_config as t
    cross join top_shap_driver as s
    where p.fraud_probability >= t.optimized_threshold
)

select *
from joined
