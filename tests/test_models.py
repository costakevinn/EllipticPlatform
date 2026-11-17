"""Tests for the models."""

import torch

from models import MLP, GraphSAGE, neighbor_mean


def test_neighbor_mean_by_hand():
    x = torch.tensor([[1.0], [0.0], [3.0]])
    edges = torch.tensor([[0, 2], [1, 1]])
    expected = torch.tensor([[0.0], [2.0], [0.0]])
    assert torch.equal(neighbor_mean(x, edges), expected)


def test_node_without_neighbors_gets_zero():
    x = torch.tensor([[5.0], [7.0]])
    edges = torch.tensor([[0], [1]])
    result = neighbor_mean(x, edges)
    assert result[0].item() == 0.0


def test_models_return_one_score_per_node():
    x = torch.randn(6, 10)
    edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
    assert MLP(10)(x, edges).shape == (6,)
    assert GraphSAGE(10)(x, edges).shape == (6,)


def test_mlp_ignores_edges():
    x = torch.randn(6, 10)
    model = MLP(10)
    no_edges = torch.zeros((2, 0), dtype=torch.long)
    edges = torch.tensor([[0, 1], [1, 2]])
    assert torch.equal(model(x, edges), model(x, no_edges))


def test_graphsage_does_not_mix_disconnected_nodes():
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    edges = torch.tensor([[0, 2], [1, 3]])
    model = GraphSAGE(5)
    before = model(x, edges)
    x_changed = x.clone()
    x_changed[0:2] += 10.0
    after = model(x_changed, edges)
    assert torch.allclose(before[2:], after[2:])
