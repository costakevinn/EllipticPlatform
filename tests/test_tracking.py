import mlflow
import pytest
import torch

import tracking
from metrics import GROUPS
from models import MLP


@pytest.fixture
def tracked(tmp_path, monkeypatch):
    """Point MLflow at a temporary folder so tests never touch data/mlflow."""
    monkeypatch.setattr(tracking, "TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(tracking, "ARTIFACT_DIR", tmp_path / "artifacts")
    tracking.setup()
    yield mlflow.MlflowClient()
    mlflow.end_run()


def tiny_data():
    return {
        "x": torch.randn(4, 3),
        "y": torch.tensor([1, 0, 1, 0]),
        "time_step": torch.tensor([35, 42, 43, 49]),
        "test_mask": torch.tensor([True, True, True, True]),
    }


def test_start_run_logs_params(tracked):
    with tracking.start_run("mlp", 7, {"epochs": 3}, tiny_data()) as run:
        run_id = run.info.run_id
    saved = tracked.get_run(run_id)
    assert saved.info.run_name == "mlp-seed7"
    assert saved.data.params == {
        "model": "mlp",
        "seed": "7",
        "in_features": "3",
        "epochs": "3",
    }


def test_log_results_logs_metrics_and_files(tracked):
    data = tiny_data()
    pred = torch.tensor([1, 0, 0, 1])
    with tracking.start_run("mlp", 0, {}, data) as run:
        run_id = run.info.run_id
        tracking.log_results(MLP(3), pred, data, train_seconds=1.5)
    metrics = tracked.get_run(run_id).data.metrics
    assert metrics["train_seconds"] == 1.5
    assert "f1_by_time_step" in metrics
    for group in GROUPS:
        assert f"f1_{group}" in metrics
    files = sorted(f.path for f in tracked.list_artifacts(run_id))
    assert files == ["by_time_step.csv", "model.pt"]


def test_setup_points_to_temp_dir(tracked, tmp_path):
    tracking.setup()
    experiment = tracked.get_experiment_by_name(tracking.EXPERIMENT)
    assert experiment.artifact_location == (tmp_path / "artifacts").resolve().as_uri()
