"""Train a model on the Gold training nodes and evaluate it by time_step."""

import sys
import time

import torch
from torch import nn

from load_gold import load_gold
from metrics import by_time_step, precision_recall_f1
from models import MLP, GraphSAGE
from tracking import log_results, setup, start_run

EPOCHS = 200
LEARNING_RATE = 1e-3
HIDDEN = 64
MODELS = {"mlp": MLP, "sage": GraphSAGE}


def train_model(name, data, seed=42, verbose=True):
    torch.manual_seed(seed)
    model = MODELS[name](data["x"].shape[1], HIDDEN)

    mask = data["train_mask"]
    y = data["y"][mask].float()
    pos_weight = (y == 0).sum() / (y == 1).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        logits = model(data["x"], data["edge_index"])
        loss = loss_fn(logits[mask], y)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 20 == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}")
    return model


def predict(model, data):
    model.eval()
    with torch.no_grad():
        return (model(data["x"], data["edge_index"]) > 0).long()


def evaluate(pred, data):
    mask = data["test_mask"]
    y_test, pred_test = data["y"][mask], pred[mask]
    p, r, f1 = precision_recall_f1(y_test, pred_test)
    print(f"test overall: precision {p:.3f}  recall {r:.3f}  f1 {f1:.3f}")

    print("time_step  illicit  precision  recall     f1")
    for step, n, p, r, f1 in by_time_step(y_test, pred_test, data["time_step"][mask]):
        print(f"{step:9d}  {n:7d}  {p:9.3f}  {r:6.3f}  {f1:5.3f}")


def train_logged(name, data, seed=42, verbose=True):
    """Train inside an open MLflow run, so time, CPU and RAM are measured."""
    params = {"epochs": EPOCHS, "learning_rate": LEARNING_RATE, "hidden": HIDDEN}
    with start_run(name, seed, params, data):
        started = time.perf_counter()
        model = train_model(name, data, seed=seed, verbose=verbose)
        train_seconds = time.perf_counter() - started
        pred = predict(model, data)
        log_results(model, pred, data, train_seconds)
    return model, pred


if __name__ == "__main__":
    name = sys.argv[1]
    seed = 42
    data = load_gold()
    if "--log" in sys.argv:
        setup()
        _, pred = train_logged(name, data, seed)
    else:
        model = train_model(name, data, seed=seed)
        pred = predict(model, data)
    evaluate(pred, data)
