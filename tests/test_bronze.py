"""Tests for the Bronze layer (files written by src/build_bronze.py)."""

from pathlib import Path

import pandas as pd
import pytest

BRONZE_DIR = Path("data/bronze/elliptic")


@pytest.fixture(scope="module")
def classes():
    return pd.read_parquet(BRONZE_DIR / "classes.parquet")


def test_classes_columns(classes):
    assert list(classes.columns) == ["tx_id", "class"]


def test_class_values(classes):
    assert set(classes["class"]) == {"unknown", "1", "2"}


@pytest.fixture(scope="module")
def edges():
    return pd.read_parquet(BRONZE_DIR / "edges.parquet")


@pytest.fixture(scope="module")
def features():
    return pd.read_parquet(BRONZE_DIR / "features.parquet")


def test_classes_ids_unique(classes):
    assert classes["tx_id"].is_unique


def test_edges_columns(edges):
    assert list(edges.columns) == ["tx_from", "tx_to"]


def test_features_columns(features):
    assert len(features.columns) == 167
    assert list(features.columns[:3]) == ["tx_id", "time_step", "local_1"]
    assert features.columns[-1] == "agg_72"


def test_features_time_steps(features):
    assert set(features["time_step"]) == set(range(1, 50))


def test_features_ids_unique(features):
    assert features["tx_id"].is_unique


def test_features_and_classes_have_same_ids(features, classes):
    assert set(features["tx_id"]) == set(classes["tx_id"])


def test_edges_ids_exist_in_classes(edges, classes):
    assert edges["tx_from"].isin(classes["tx_id"]).all()
    assert edges["tx_to"].isin(classes["tx_id"]).all()
