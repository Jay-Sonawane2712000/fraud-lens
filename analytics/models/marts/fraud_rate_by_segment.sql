with features as (
    select *
    from {{ ref('stg_transactions_features') }}
),

segmented as (
    select
        case
            when coalesce(has_identity_info, 0) = 0 then 'no_identity_info'
            when coalesce(device_type_missing, 0) = 1
                and coalesce(device_info_missing, 0) = 1
                then 'device_type_and_info_missing'
            when coalesce(device_type_missing, 0) = 1 then 'device_type_missing'
            when coalesce(device_info_missing, 0) = 1 then 'device_info_missing'
            else 'identity_and_device_present'
        end as identity_device_category,
        case
            when coalesce(email_domain_match, 0) = 1 then 'email_domain_match'
            else 'email_domain_mismatch_or_missing'
        end as email_domain_risk_tier,
        case
            when TransactionAmt is null then 'unknown_amount'
            when TransactionAmt < 50 then 'amount_under_50'
            when TransactionAmt < 200 then 'amount_50_to_199'
            when TransactionAmt < 500 then 'amount_200_to_499'
            else 'amount_500_plus'
        end as transaction_amount_band,
        case
            when transaction_hour between 0 and 5 then 'late_night_00_05'
            when transaction_hour between 6 and 11 then 'morning_06_11'
            when transaction_hour between 12 and 17 then 'afternoon_12_17'
            when transaction_hour between 18 and 23 then 'evening_18_23'
            else 'unknown_hour'
        end as time_window,
        isFraud
    from features
),

aggregated as (
    select
        concat(
            identity_device_category, '|',
            email_domain_risk_tier, '|',
            transaction_amount_band, '|',
            time_window
        ) as segment_key,
        identity_device_category,
        email_domain_risk_tier,
        transaction_amount_band,
        time_window,
        count(*) as transaction_count,
        sum(isFraud) as fraud_count,
        cast(sum(isFraud) as double) / count(*) as fraud_rate
    from segmented
    group by
        identity_device_category,
        email_domain_risk_tier,
        transaction_amount_band,
        time_window
)

select *
from aggregated
where transaction_count >= {{ var('min_segment_size') }}
