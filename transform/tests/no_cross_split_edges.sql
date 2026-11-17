-- Gold: no edge may connect a train transaction to a test transaction (leakage guard)
select e.*
from {{ ref('graph_edges') }} as e
join {{ ref('transactions') }} as t on e.tx_to = t.tx_id
where e.split <> t.split
