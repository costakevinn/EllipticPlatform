select
    e.tx_from,
    e.tx_to,
    t.time_step,
    t.split
from {{ ref('edges') }} as e
join {{ ref('transactions') }} as t on e.tx_from = t.tx_id
