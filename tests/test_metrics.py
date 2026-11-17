"""Tests for the illicit-class metrics."""

import torch

from metrics import by_time_step, precision_recall_f1


def test_precision_recall_f1_by_hand():
    y_true = torch.tensor([1, 1, 0, 0])
    y_pred = torch.tensor([1, 0, 1, 0])
    assert precision_recall_f1(y_true, y_pred) == (0.5, 0.5, 0.5)


def test_no_illicit_predicted_gives_zero():
    y_true = torch.tensor([1, 0])
    y_pred = torch.tensor([0, 0])
    assert precision_recall_f1(y_true, y_pred) == (0.0, 0.0, 0.0)


def test_by_time_step_splits_per_step():
    y_true = torch.tensor([1, 0, 1, 1])
    y_pred = torch.tensor([1, 0, 0, 0])
    steps = torch.tensor([35, 35, 36, 36])
    rows = by_time_step(y_true, y_pred, steps)
    assert rows == [(35, 1, 1.0, 1.0, 1.0), (36, 2, 0.0, 0.0, 0.0)]
