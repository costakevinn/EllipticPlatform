"""Load the Gold tables as PyTorch tensors."""

import duckdb
import torch

DB_PATH = "data/warehouse/elliptic.duckdb"
NOT_FEATURES = ["tx_id", "time_step", "split", "label"]
DEGREE_COLS = ["out_degree", "in_degree"]
LABEL_CODES = {"licit": 0, "illicit": 1}


def load_gold():
    con = duckdb.connect(DB_PATH, read_only=True)
    nodes = con.sql("SELECT * FROM main_gold.transactions ORDER BY tx_id").df()
    edges = con.sql("SELECT tx_from, tx_to FROM main_gold.graph_edges").df()
    con.close()

    feature_cols = [c for c in nodes.columns if c not in NOT_FEATURES]
    x = torch.tensor(nodes[feature_cols].to_numpy(), dtype=torch.float32)
    degree_idx = [feature_cols.index(c) for c in DEGREE_COLS]
    x[:, degree_idx] = torch.log1p(x[:, degree_idx])

    codes = nodes["label"].map(LABEL_CODES).fillna(-1).astype(int)
    y = torch.tensor(codes.to_numpy(), dtype=torch.long)

    position = {tx_id: i for i, tx_id in enumerate(nodes["tx_id"])}
    source = edges["tx_from"].map(position).to_list()
    target = edges["tx_to"].map(position).to_list()
    edge_index = torch.tensor([source, target], dtype=torch.long)

    is_train = torch.tensor((nodes["split"] == "train").to_numpy())
    known = y >= 0
    return {
        "x": x,
        "y": y,
        "edge_index": edge_index,
        "time_step": torch.tensor(nodes["time_step"].to_numpy()),
        "tx_id": torch.tensor(nodes["tx_id"].to_numpy()),
        "is_test": ~is_train,
        "train_mask": is_train & known,
        "test_mask": ~is_train & known,
    }


if __name__ == "__main__":
    data = load_gold()
    for name, tensor in data.items():
        print(name, tuple(tensor.shape), tensor.dtype)
    print("labeled train:", int(data["train_mask"].sum()))
    print("labeled test:", int(data["test_mask"].sum()))
