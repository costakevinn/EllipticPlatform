"""FastAPI app: serves the predictions and metrics stored in PostgreSQL."""

import os
from typing import Annotated

import mlflow
import psycopg
from fastapi import Depends, FastAPI, HTTPException, Path, Query
from psycopg.rows import dict_row

from schemas import (
    Alerts,
    Health,
    Metrics,
    Prediction,
    Run,
    StepMetrics,
    StreamAlert,
    StreamAlerts,
)
from tracking import EXPERIMENT, TRACKING_URI

app = FastAPI(title="EllipticPlatform API")

VERDICTS = {0: "licit", 1: "illicit"}


def get_db():
    conn = psycopg.connect(
        host=os.environ["POSTGRES_HOST"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
        row_factory=dict_row,
    )
    try:
        yield conn
    finally:
        conn.close()


Db = Annotated[psycopg.Connection, Depends(get_db)]


def to_prediction(row):
    label = row["label"]
    return Prediction(
        run_id=row["run_id"],
        tx_id=row["tx_id"],
        time_step=row["time_step"],
        score=row["score"],
        predicted=VERDICTS[row["predicted"]],
        label=None if label is None else VERDICTS[label],
    )


@app.get("/health", response_model=Health)
def health(db: Db):
    db.execute("select 1")
    return Health(status="ok", database="ok")


@app.get("/transactions/{tx_id}", response_model=Prediction)
def get_transaction(tx_id: int, run_id: str, db: Db):
    row = db.execute(
        "select run_id, tx_id, time_step, score, predicted, label "
        "from predictions where run_id = %s and tx_id = %s",
        (run_id, tx_id),
    ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail="transaction not found for this run"
        )
    return to_prediction(row)


@app.get("/time-steps/{time_step}/alerts", response_model=Alerts)
def get_alerts(
    time_step: Annotated[int, Path(ge=1, le=49)],
    run_id: str,
    db: Db,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    has_data = db.execute(
        "select 1 from predictions where run_id = %s and time_step = %s limit 1",
        (run_id, time_step),
    ).fetchone()
    if has_data is None:
        raise HTTPException(
            status_code=404, detail="no predictions for this run and step"
        )
    rows = db.execute(
        "select run_id, tx_id, time_step, score, predicted, label "
        "from predictions "
        "where run_id = %s and time_step = %s and predicted = 1 "
        "order by score desc, tx_id limit %s",
        (run_id, time_step, limit),
    ).fetchall()
    return Alerts(
        run_id=run_id,
        time_step=time_step,
        alerts=[to_prediction(row) for row in rows],
    )


@app.get("/metrics/by-time-step", response_model=Metrics)
def get_metrics_by_time_step(run_id: str, db: Db):
    rows = db.execute(
        "select time_step, n_illicit, "
        "precision_illicit as precision, recall_illicit as recall, f1_illicit as f1 "
        "from metrics_by_time_step where run_id = %s order by time_step",
        (run_id,),
    ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="no metrics for this run")
    return Metrics(run_id=run_id, steps=[StepMetrics(**row) for row in rows])


@app.get("/runs", response_model=list[Run])
def get_runs():
    mlflow.set_tracking_uri(TRACKING_URI)
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT], output_format="list")
    complete = [r for r in runs if "train_seconds" in r.data.metrics]
    return [
        Run(
            run_id=r.info.run_id,
            name=r.info.run_name,
            model=r.data.params["model"],
            seed=int(r.data.params["seed"]),
            train_seconds=r.data.metrics["train_seconds"],
            f1_all=r.data.metrics["f1_all"],
            f1_35_42=r.data.metrics["f1_35_42"],
            f1_43_49=r.data.metrics["f1_43_49"],
        )
        for r in sorted(complete, key=lambda r: r.info.run_name)
    ]


@app.get("/stream/time-steps/{time_step}/alerts", response_model=StreamAlerts)
def get_stream_alerts(
    time_step: Annotated[int, Path(ge=1, le=49)],
    run_id: str,
    db: Db,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    known = db.execute(
        "select 1 from alerts where run_id = %s limit 1", (run_id,)
    ).fetchone()
    if known is None:
        raise HTTPException(status_code=404, detail="no streamed alerts for this run")
    rows = db.execute(
        "select run_id, tx_id, time_step, score, scored_at from alerts "
        "where run_id = %s and time_step = %s order by score desc, tx_id limit %s",
        (run_id, time_step, limit),
    ).fetchall()
    return StreamAlerts(
        run_id=run_id,
        time_step=time_step,
        alerts=[StreamAlert(**row) for row in rows],
    )
