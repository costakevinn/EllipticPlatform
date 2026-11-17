-- EDA 3.2: no repeated edges
select tx_from, tx_to, count(*) as n
from {{ ref('edges') }}
group by tx_from, tx_to
having count(*) > 1
