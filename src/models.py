"""Models for the Elliptic transactions."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, edge_index=None):
        return self.net(x).squeeze(-1)


def neighbor_mean(x, edge_index):
    source, target = edge_index
    total = torch.zeros_like(x).index_add_(0, target, x[source])
    count = x.new_zeros(x.shape[0]).index_add_(0, target, x.new_ones(source.shape[0]))
    return total / count.clamp(min=1).unsqueeze(-1)


class SAGELayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.neighbor_linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, edge_index):
        return self.self_linear(x) + self.neighbor_linear(neighbor_mean(x, edge_index))


class GraphSAGE(nn.Module):
    def __init__(self, in_features, hidden=64):
        super().__init__()
        self.layer1 = SAGELayer(in_features, hidden)
        self.layer2 = SAGELayer(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        h = torch.relu(self.layer1(x, edge_index))
        h = torch.relu(self.layer2(h, edge_index))
        return self.out(h).squeeze(-1)
