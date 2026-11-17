"""Consume transactions from Kafka, score them with an MLflow MLP run, store alerts in PostgreSQL."""

import argparse
import json
import os
import time

import mlflow
import torch
from confluent_kafka import Consumer

from tracking import EXPERIMENT, TRACKING_URI
from write_predictions import connect, load_model

TOPIC = "transactions"
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
GROUP = "scorer"
BATCH = 500


def find_run(name):
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="list")
    matches = [r for r in runs if r.info.run_name == name]
    if not matches:
        raise SystemExit(f"no MLflow run named {name}")
    return matches[0]


def score(model, run_id, batch):
    x = torch.tensor([m["features"] for m in batch], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
    scores = torch.sigmoid(logits).tolist()
    return [
        (run_id, m["tx_id"], m["time_step"], s)
        for m, s, logit in zip(batch, scores, logits.tolist(), strict=True)
        if logit > 0
    ]


def store(conn, rows):
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "insert into alerts (run_id, tx_id, time_step, score) values (%s, %s, %s, %s) "
            "on conflict (run_id, tx_id) do nothing",
            rows,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Score Kafka transactions and store alerts"
    )
    parser.add_argument(
        "--run", default="mlp-seed42", help="MLflow run name of the MLP to use"
    )
    parser.add_argument(
        "--idle", type=float, default=20.0, help="stop after N idle seconds"
    )
    parser.add_argument("--group", default=GROUP)
    args = parser.parse_args()

    run = find_run(args.run)
    if run.data.params["model"] != "mlp":
        raise SystemExit(
            "the streaming consumer needs an MLP run (GraphSAGE needs the graph)"
        )
    model = load_model(run)
    conn = connect()
    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "group.id": args.group,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([TOPIC])

    seen = alerts = 0
    last_message = time.monotonic()
    try:
        while time.monotonic() - last_message < args.idle:
            messages = consumer.consume(num_messages=BATCH, timeout=1.0)
            batch = []
            for message in messages:
                if message.error():
                    raise RuntimeError(message.error())
                batch.append(json.loads(message.value()))
            if not batch:
                continue
            last_message = time.monotonic()
            rows = score(model, run.info.run_id, batch)
            store(conn, rows)
            consumer.commit(asynchronous=False)
            seen += len(batch)
            alerts += len(rows)
            print(f"step {batch[-1]['time_step']}: scored {seen}, alerts {alerts}")
    finally:
        consumer.close()
        conn.close()
    print(f"done: scored {seen}, alerts {alerts}")


if __name__ == "__main__":
    main()
