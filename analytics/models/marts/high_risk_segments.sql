with segments as (
    select *
    from {{ ref('fraud_rate_by_segment') }}
),

overall as (
    select
        cast(sum(isFraud) as double) / count(*) as overall_base_fraud_rate,
        sum(isFraud) as total_fraud_count
    from {{ ref('stg_transactions_features') }}
),

high_risk as (
    select
        s.segment_key,
        s.identity_device_category,
        s.email_domain_risk_tier,
        s.transaction_amount_band,
        s.time_window,
        s.transaction_count,
        s.fraud_count,
        s.fraud_rate,
        o.overall_base_fraud_rate,
        s.fraud_rate / o.overall_base_fraud_rate as fraud_rate_lift,
        cast(s.fraud_count as double) / nullif(o.total_fraud_count, 0) as fraud_volume_share
    from segments as s
    cross join overall as o
    where s.fraud_rate > o.overall_base_fraud_rate
),

ranked as (
    select
        *,
        row_number() over (
            order by fraud_count desc, fraud_rate_lift desc, transaction_count desc
        ) as risk_rank
    from high_risk
)

select *
from ranked
