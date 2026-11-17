-- EDA 4.3: time_step goes from 1 to 49
select *
from {{ ref('features') }}
where time_step < 1 or time_step > 49
