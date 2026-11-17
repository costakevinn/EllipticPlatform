-- EDA 5.3: every edge connects two transactions in the same time step
select e.tx_from, e.tx_to
from {{ ref('edges') }} as e
join {{ ref('features') }} as f1 on e.tx_from = f1.tx_id
join {{ ref('features') }} as f2 on e.tx_to = f2.tx_id
where f1.time_step <> f2.time_step
