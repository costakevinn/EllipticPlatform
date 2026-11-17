"""Score the test split with every MLflow run's model and store it in PostgreSQL."""

import os
import tempfile

import mlflow
import psycopg
import torch

from load_gold import load_gold
from metrics import by_time_step, f1_in_steps
from tracking import EXPERIMENT, TRACKING_URI
from train import MODELS

FIRST_TEST_STEP = 35
LAST_TEST_STEP = 49


def connect():
    return psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )


def load_model(run):
    params = run.data.params
    model = MODELS[params["model"]](int(params["in_features"]), int(params["hidden"]))
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id, artifact_path="model.pt", dst_path=tmp
        )
        model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def write_run(conn, run, data):
    run_id = run.info.run_id
    model = load_model(run)
    with torch.no_grad():
        logits = model(data["x"], data["edge_index"])
    pred = (logits > 0).long()

    in_test = data["is_test"]
    scores = torch.sigmoid(logits[in_test]).tolist()
    predicted = pred[in_test].tolist()
    tx_ids = data["tx_id"][in_test].tolist()
    steps = data["time_step"][in_test].tolist()
    labels = [None if v < 0 else v for v in data["y"][in_test].tolist()]

    labeled = data["test_mask"]
    metric_rows = by_time_step(
        data["y"][labeled], pred[labeled], data["time_step"][labeled]
    )

    with conn.transaction():
        conn.execute("DELETE FROM predictions WHERE run_id = %s", (run_id,))
        conn.execute("DELETE FROM metrics_by_time_step WHERE run_id = %s", (run_id,))
        with conn.cursor() as cur:
            sql = (
                "COPY predictions (run_id, tx_id, time_step, score, predicted, label) "
                "FROM STDIN"
            )
            with cur.copy(sql) as copy:
                for row in zip(tx_ids, steps, scores, predicted, labels):
                    copy.write_row((run_id, *row))
            cur.executemany(
                "INSERT INTO metrics_by_time_step (run_id, time_step, n_illicit, "
                "precision_illicit, recall_illicit, f1_illicit) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (run_id, int(s), int(n), float(p), float(r), float(f))
                    for s, n, p, r, f in metric_rows
                ],
            )
    return len(tx_ids), f1_in_steps(pred, data, FIRST_TEST_STEP, LAST_TEST_STEP)


STALE_TABLES = ("predictions", "metrics_by_time_step", "alerts")


def remove_stale(conn, run_ids):
    """Delete rows of runs that no longer exist in MLflow; count per table."""
    removed = {}
    with conn.transaction():
        for table in STALE_TABLES:
            cur = conn.execute(
                f"DELETE FROM {table} WHERE run_id <> ALL(%s)", (list(run_ids),)
            )
            removed[table] = cur.rowcount
    return removed


def main():
    data = load_gold()
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="list")
    if not runs:
        raise SystemExit("no MLflow runs found; refusing to clean the database")
    with connect() as conn:
        removed = remove_stale(conn, [run.info.run_id for run in runs])
        print("removed stale rows:", ", ".join(f"{t} {n}" for t, n in removed.items()))
        for run in sorted(runs, key=lambda r: r.info.run_name):
            n_rows, f1 = write_run(conn, run, data)
            logged = run.data.metrics["f1_all"]
            name = run.info.run_name
            print(
                f"{name:11s} rows {n_rows}  f1_all rebuilt {f1:.3f}  mlflow {logged:.3f}"
            )


if __name__ == "__main__":
    main()
