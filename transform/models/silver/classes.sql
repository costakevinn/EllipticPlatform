select
    tx_id,
    case class
        when '1' then 'illicit'
        when '2' then 'licit'
        else null
    end as label
from read_parquet('data/bronze/elliptic/classes.parquet')
