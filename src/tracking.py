"""Log training runs to MLflow."""

import tempfile
from pathlib import Path

import mlflow
import torch

from metrics import GROUPS, by_time_step, f1_in_steps

TRACKING_URI = "sqlite:///data/mlflow/mlflow.db"
ARTIFACT_DIR = Path("data/mlflow/artifacts")
EXPERIMENT = "elliptic-mlp-vs-graphsage"
SAMPLING_SECONDS = 5


def setup():
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        location = ARTIFACT_DIR.resolve().as_uri()
        mlflow.create_experiment(EXPERIMENT, artifact_location=location)
    mlflow.set_experiment(EXPERIMENT)
    mlflow.enable_system_metrics_logging()
    mlflow.set_system_metrics_sampling_interval(SAMPLING_SECONDS)


def start_run(name, seed, params, data):
    """Open a run and log its parameters; use it as a `with` block around training."""
    run = mlflow.start_run(run_name=f"{name}-seed{seed}")
    all_params = {"model": name, "seed": seed, "in_features": data["x"].shape[1]}
    mlflow.log_params({**all_params, **params})
    return run


def log_results(model, pred, data, train_seconds):
    """Log metrics and files of the open run, after training."""
    mlflow.log_metric("train_seconds", train_seconds)
    for group, (first, last) in GROUPS.items():
        mlflow.log_metric(f"f1_{group}", f1_in_steps(pred, data, first, last))

    mask = data["test_mask"]
    rows = by_time_step(data["y"][mask], pred[mask], data["time_step"][mask])
    for step, _, _, _, f1 in rows:
        mlflow.log_metric("f1_by_time_step", f1, step=step)

    lines = ["time_step,illicit,precision,recall,f1"]
    lines += [f"{s},{n},{p:.4f},{r:.4f},{f:.4f}" for s, n, p, r, f in rows]
    mlflow.log_text("\n".join(lines) + "\n", "by_time_step.csv")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save(model.state_dict(), path)
        mlflow.log_artifact(str(path))
