"""Checks on the elliptic_pipeline DAG.

Run it inside the Airflow scheduler container, where Airflow is installed:
docker compose --profile orchestrate exec -T airflow-scheduler python - < tests/check_dag.py
"""

import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    "elliptic_pipeline", "/opt/airflow/dags/elliptic_pipeline.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
dag = module.dag
tasks = dag.task_dict

EXPECTED_TASKS = {
    "ingest_raw",
    "build_bronze",
    "silver",
    "gold",
    "train_mlp",
    "train_sage",
    "start_postgres",
    "write_predictions",
    "start_kafka",
    "recreate_topic",
    "replay_consume",
    "replay_produce",
    "verify_replay",
    "stop_kafka",
    "stop_postgres",
}
EDGES = [
    ("ingest_raw", "build_bronze"),
    ("build_bronze", "silver"),
    ("silver", "gold"),
    ("gold", "train_mlp"),
    ("train_mlp", "train_sage"),
    ("train_sage", "start_postgres"),
    ("start_postgres", "write_predictions"),
    ("write_predictions", "start_kafka"),
    ("start_kafka", "recreate_topic"),
    ("recreate_topic", "replay_consume"),
    ("recreate_topic", "replay_produce"),
    ("replay_consume", "verify_replay"),
    ("replay_produce", "verify_replay"),
    ("verify_replay", "stop_kafka"),
    ("verify_replay", "stop_postgres"),
]
MEMORY = {
    "ingest_raw": "2g",
    "build_bronze": "1536m",
    "silver": "1g",
    "gold": "1g",
    "train_mlp": "2560m",
    "train_sage": "3g",
    "write_predictions": "2g",
    "recreate_topic": "1g",
    "replay_consume": "1g",
    "replay_produce": "2560m",
    "verify_replay": "1g",
}
TRAINING = {"train_mlp", "train_sage"}
HEAVY = TRAINING | {"write_predictions"}
SECRETS = {"KAGGLE_API_TOKEN", "POSTGRES_PASSWORD"}


def reachable(start):
    """Every task that runs after `start`, directly or not."""
    seen, pending = set(), [start]
    while pending:
        for nxt in tasks[pending.pop()].downstream_task_ids:
            if nxt not in seen:
                seen.add(nxt)
                pending.append(nxt)
    return seen


def test_task_ids():
    assert set(tasks) == EXPECTED_TASKS, set(tasks) ^ EXPECTED_TASKS


def test_dependencies():
    for before, after in EDGES:
        assert after in tasks[before].downstream_task_ids, f"{before} >> {after}"


def test_replay_tasks_are_parallel():
    assert "replay_produce" not in reachable("replay_consume")
    assert "replay_consume" not in reachable("replay_produce")


def test_services_setup_and_teardown():
    for name in ("start_postgres", "start_kafka"):
        assert tasks[name].is_setup, name
    for name in ("stop_postgres", "stop_kafka"):
        assert tasks[name].is_teardown, name
        assert name in reachable("verify_replay"), name


def test_retries():
    for name, task in tasks.items():
        expected = 0 if name in TRAINING else 2
        assert task.retries == expected, f"{name}: {task.retries}"


def test_pools():
    for name, task in tasks.items():
        expected = "heavy" if name in HEAVY else "default_pool"
        assert task.pool == expected, f"{name}: {task.pool}"


def test_memory_limits():
    for name, limit in MEMORY.items():
        assert tasks[name].mem_limit == limit, f"{name}: {tasks[name].mem_limit}"


def test_thread_limits():
    for name in HEAVY:
        assert tasks[name].environment.get("OMP_NUM_THREADS") == "4", name
    assert tasks["replay_consume"].environment.get("OMP_NUM_THREADS") == "2"


def test_secrets_stay_private():
    for name in MEMORY:
        assert not SECRETS & set(tasks[name].environment), name


def test_replay_uses_a_group_per_run():
    command = tasks["replay_consume"].command
    assert "--group replay-{{ run_id" in command, command
    assert "--idle 60" in command, command


def run():
    failed = 0
    names = [n for n in list(globals()) if n.startswith("test_")]
    for name in names:
        try:
            globals()[name]()
            print("ok   ", name)
        except AssertionError as exc:
            failed += 1
            print("FAIL ", name, "-", exc)
    print(f"{len(names) - failed} of {len(names)} checks passed")
    sys.exit(1 if failed else 0)


run()
