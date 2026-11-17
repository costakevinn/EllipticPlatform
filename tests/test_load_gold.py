"""Tests for the tensors returned by load_gold."""

import pytest

from load_gold import load_gold


@pytest.fixture(scope="module")
def data():
    return load_gold()


def test_one_row_per_transaction(data):
    n = data["x"].shape[0]
    assert n == 203_769
    assert data["y"].shape == (n,)
    assert data["time_step"].shape == (n,)


def test_feature_count(data):
    assert data["x"].shape[1] == 167


def test_edge_index_shape(data):
    assert data["edge_index"].shape == (2, 234_355)


def test_edges_point_to_existing_nodes(data):
    n = data["x"].shape[0]
    assert data["edge_index"].min() >= 0
    assert data["edge_index"].max() < n


def test_label_codes(data):
    assert set(data["y"].unique().tolist()) == {-1, 0, 1}


def test_masks_only_have_labeled_nodes(data):
    assert (data["y"][data["train_mask"]] >= 0).all()
    assert (data["y"][data["test_mask"]] >= 0).all()


def test_masks_do_not_overlap(data):
    assert not (data["train_mask"] & data["test_mask"]).any()


def test_mask_sizes(data):
    assert int(data["train_mask"].sum()) == 29_894
    assert int(data["test_mask"].sum()) == 16_670


def test_split_follows_time(data):
    steps = data["time_step"]
    assert steps[data["train_mask"]].max() <= 34
    assert steps[data["test_mask"]].min() >= 35


def test_tx_id_sorted_and_unique(data):
    tx_id = data["tx_id"]
    assert tx_id.shape == data["y"].shape
    assert bool((tx_id[1:] > tx_id[:-1]).all())


def test_is_test_covers_whole_test_split(data):
    assert int(data["is_test"].sum()) == 67_504


def test_is_test_follows_time(data):
    assert bool((data["is_test"] == (data["time_step"] >= 35)).all())


def test_is_test_includes_unlabeled(data):
    assert not bool((data["test_mask"] & ~data["is_test"]).any())
    assert int(data["is_test"].sum()) > int(data["test_mask"].sum())
