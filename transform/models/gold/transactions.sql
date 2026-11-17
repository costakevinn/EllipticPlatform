with out_degrees as (
    select tx_from as tx_id, count(*) as out_degree
    from {{ ref('edges') }}
    group by tx_from
),

in_degrees as (
    select tx_to as tx_id, count(*) as in_degree
    from {{ ref('edges') }}
    group by tx_to
)

select
    f.tx_id,
    f.time_step,
    case when f.time_step <= 34 then 'train' else 'test' end as split,
    c.label,
    coalesce(o.out_degree, 0) as out_degree,
    coalesce(i.in_degree, 0) as in_degree,
    f.* exclude (tx_id, time_step)
from {{ ref('features') }} as f
join {{ ref('classes') }} as c on f.tx_id = c.tx_id
left join out_degrees as o on f.tx_id = o.tx_id
left join in_degrees as i on f.tx_id = i.tx_id
