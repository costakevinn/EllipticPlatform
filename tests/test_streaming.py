import json

import pytest
import torch

from consumer import score
from producer import TOPIC, publish_step


class FakeProducer:
    def __init__(self):
        self.sent = []

    def produce(self, topic, key, value, on_delivery):
        self.sent.append((topic, key, json.loads(value)))

    def poll(self, timeout):
        pass


def make_data():
    return {
        "x": torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32),
        "tx_id": torch.tensor([10, 20, 30]),
        "time_step": torch.tensor([35, 36, 35]),
    }


def test_publish_step_sends_only_that_step():
    producer = FakeProducer()
    count = publish_step(producer, make_data(), 35, callback=None)
    assert count == 2
    assert [m[2]["tx_id"] for m in producer.sent] == [10, 30]
    assert all(m[0] == TOPIC and m[1] == "35" for m in producer.sent)


def test_publish_step_features_roundtrip_exactly():
    producer = FakeProducer()
    data = make_data()
    publish_step(producer, data, 36, callback=None)
    features = torch.tensor(producer.sent[0][2]["features"], dtype=torch.float32)
    assert torch.equal(features, data["x"][1])


def test_score_keeps_only_positive_logits():
    batch = [
        {"tx_id": 1, "time_step": 40, "features": [2.0, 0.0]},
        {"tx_id": 2, "time_step": 40, "features": [-1.0, 0.0]},
        {"tx_id": 3, "time_step": 41, "features": [0.5, 0.0]},
    ]
    rows = score(lambda x: x[:, 0], "run", batch)
    assert [row[1] for row in rows] == [1, 3]
    assert rows[0][0] == "run"
    assert rows[1][2] == 41
    assert rows[0][3] == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())


def test_score_is_empty_without_suspicious_transactions():
    batch = [{"tx_id": 1, "time_step": 40, "features": [-3.0, 0.0]}]
    assert score(lambda x: x[:, 0], "run", batch) == []
