select
    tx_from,
    tx_to
from read_parquet('data/bronze/elliptic/edges.parquet')
