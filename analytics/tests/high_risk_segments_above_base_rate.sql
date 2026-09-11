select *
from {{ ref('high_risk_segments') }}
where fraud_rate <= overall_base_fraud_rate
