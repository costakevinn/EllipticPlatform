import pytest
from fastapi.testclient import TestClient

from api import app

KNOWN_TX = 72359149


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def run_id(client):
    runs = client.get("/runs").json()
    return next(r["run_id"] for r in runs if r["name"] == "mlp-seed42")


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_runs_lists_complete_runs(client):
    runs = client.get("/runs").json()
    names = [r["name"] for r in runs]
    assert names == sorted(names)
    assert "mlp-seed42" in names
    assert all(r["train_seconds"] > 0 for r in runs)


def test_transaction_found(client, run_id):
    response = client.get(f"/transactions/{KNOWN_TX}", params={"run_id": run_id})
    body = response.json()
    assert response.status_code == 200
    assert body["tx_id"] == KNOWN_TX
    assert body["predicted"] == "illicit"
    assert body["label"] is None
    assert 0 <= body["score"] <= 1


def test_transaction_not_found(client, run_id):
    response = client.get("/transactions/1", params={"run_id": run_id})
    assert response.status_code == 404


def test_transaction_requires_run_id(client):
    assert client.get(f"/transactions/{KNOWN_TX}").status_code == 422


def test_alerts_sorted_by_score_desc(client, run_id):
    response = client.get(
        "/time-steps/42/alerts", params={"run_id": run_id, "limit": 3}
    )
    body = response.json()
    scores = [a["score"] for a in body["alerts"]]
    assert response.status_code == 200
    assert len(scores) == 3
    assert scores == sorted(scores, reverse=True)
    assert all(a["predicted"] == "illicit" for a in body["alerts"])


def test_alerts_step_without_data_is_404(client, run_id):
    assert (
        client.get("/time-steps/10/alerts", params={"run_id": run_id}).status_code
        == 404
    )


def test_alerts_invalid_parameters_are_422(client, run_id):
    assert (
        client.get("/time-steps/99/alerts", params={"run_id": run_id}).status_code
        == 422
    )
    params = {"run_id": run_id, "limit": 500}
    assert client.get("/time-steps/42/alerts", params=params).status_code == 422


def test_metrics_cover_test_steps(client, run_id):
    response = client.get("/metrics/by-time-step", params={"run_id": run_id})
    steps = [s["time_step"] for s in response.json()["steps"]]
    assert response.status_code == 200
    assert steps == list(range(35, 50))


def test_metrics_unknown_run_is_404(client):
    response = client.get("/metrics/by-time-step", params={"run_id": "nao-existe"})
    assert response.status_code == 404


def test_stream_alerts_match_batch_alerts(client, run_id):
    params = {"run_id": run_id, "limit": 3}
    stream = client.get("/stream/time-steps/42/alerts", params=params).json()
    batch = client.get("/time-steps/42/alerts", params=params).json()
    assert [a["tx_id"] for a in stream["alerts"]] == [
        a["tx_id"] for a in batch["alerts"]
    ]
    scores = [a["score"] for a in stream["alerts"]]
    assert scores == sorted(scores, reverse=True)


def test_stream_alerts_unknown_run_is_404(client):
    response = client.get(
        "/stream/time-steps/42/alerts", params={"run_id": "nao-existe"}
    )
    assert response.status_code == 404


def test_stream_alerts_invalid_step_is_422(client, run_id):
    response = client.get("/stream/time-steps/99/alerts", params={"run_id": run_id})
    assert response.status_code == 422
