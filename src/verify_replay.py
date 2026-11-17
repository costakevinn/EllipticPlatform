"""Check that the streamed alerts match the batch predictions for one MLflow run."""

import argparse
import sys

from consumer import find_run
from write_predictions import connect

TOLERANCE = 1e-5

QUERY = """
select
    count(*) filter (where a.tx_id is not null and p.tx_id is not null) as both_sides,
    count(*) filter (where p.tx_id is null) as only_stream,
    count(*) filter (where a.tx_id is null) as only_batch,
    coalesce(max(abs(a.score - p.score)), 0) as max_score_diff
from (select * from alerts where run_id = %(run)s) a
full outer join (
    select * from predictions where run_id = %(run)s and predicted = 1
) p using (tx_id)
"""


def main():
    parser = argparse.ArgumentParser(description="Compare stream alerts with batch")
    parser.add_argument("--run", default="mlp-seed42", help="MLflow run name")
    args = parser.parse_args()

    found = find_run(args.run)
    run_id = found if isinstance(found, str) else found.info.run_id

    with connect() as conn:
        both, only_stream, only_batch, max_diff = conn.execute(
            QUERY, {"run": run_id}
        ).fetchone()

    print(f"run {args.run} ({run_id})")
    print(f"in both: {both}, only stream: {only_stream}, only batch: {only_batch}")
    print(f"max score difference: {max_diff:.2e}")

    ok = both > 0 and only_stream == 0 and only_batch == 0 and max_diff <= TOLERANCE
    print("OK: stream equals batch" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
