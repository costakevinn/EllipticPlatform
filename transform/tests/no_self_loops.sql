-- EDA 3.3: no self-loops
select *
from {{ ref('edges') }}
where tx_from = tx_to
