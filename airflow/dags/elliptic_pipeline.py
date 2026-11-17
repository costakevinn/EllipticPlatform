"""Elliptic pipeline: batch (raw to predictions), then Kafka replay and check."""

import os
import time
from datetime import datetime, timedelta

import docker
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG, task
from docker.types import Mount

IMAGE = "elliptic-dev"
KAFKA_IMAGE = "apache/kafka:4.3.1"
NETWORK = "ellipticplatform_default"
PROJECT_DIR = os.environ["HOST_PROJECT_DIR"]
USER = f"{os.environ['HOST_UID']}:{os.environ['HOST_GID']}"
DBT = "--project-dir transform --profiles-dir transform"
POSTGRES = {
    "POSTGRES_HOST": "postgres",
    "POSTGRES_USER": "elliptic",
    "POSTGRES_DB": "elliptic",
}
POSTGRES_SECRET = {"POSTGRES_PASSWORD": os.environ["POSTGRES_PASSWORD"]}
# One consumer group per DAG run, so every replay starts from the first message.
GROUP = "replay-{{ run_id | replace(':', '-') | replace('+', '-') }}"
KAFKA_TOPICS = "/opt/kafka/bin/kafka-topics.sh --bootstrap-server kafka:9092"
POSTGRES_CONTAINER = "ellipticplatform-postgres-1"
KAFKA_CONTAINER = "ellipticplatform-kafka-1"
POSTGRES_READY = ["pg_isready", "-U", "elliptic"]
KAFKA_READY = [
    "/opt/kafka/bin/kafka-topics.sh",
    "--bootstrap-server",
    "kafka:9092",
    "--list",
]
READY_TIMEOUT = 120
# PyTorch uses 5 threads by default on this 10-CPU machine; cap it per stage.
THREADS_HEAVY = {"OMP_NUM_THREADS": "4"}
THREADS_LIGHT = {"OMP_NUM_THREADS": "2"}
# Pool with one slot: heavy stages of overlapping runs never run together.
HEAVY_POOL = "heavy"


@task
def start_service(container, ready_command):
    """Start a Compose container and wait until it answers."""
    box = docker.from_env().containers.get(container)
    box.start()
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        if box.exec_run(ready_command).exit_code == 0:
            return
        time.sleep(2)
    raise RuntimeError(f"{container} not ready after {READY_TIMEOUT}s")


@task
def stop_service(container):
    """Stop a Compose container."""
    docker.from_env().containers.get(container).stop(timeout=30)


def stage(
    task_id,
    command,
    mem_limit,
    environment=None,
    secrets=None,
    project=True,
    **extra,
):
    """One DockerOperator: a container with its own memory limit."""
    kwargs = {
        "task_id": task_id,
        "container_name": f"elliptic-{task_id}",
        "image": IMAGE,
        "command": command,
        "network_mode": NETWORK,
        "mem_limit": mem_limit,
        "environment": environment or {},
        "private_environment": secrets or {},
        "auto_remove": "force",
        "mount_tmp_dir": False,
    }
    if project:
        kwargs["user"] = USER
        kwargs["mounts"] = [Mount(source=PROJECT_DIR, target="/workspace", type="bind")]
    kwargs.update(extra)
    return DockerOperator(**kwargs)


with DAG(
    dag_id="elliptic_pipeline",
    start_date=datetime(2026, 11, 1),
    schedule=None,
    catchup=False,
    tags=["elliptic"],
    default_args={"retries": 2, "retry_delay": timedelta(seconds=30)},
) as dag:
    # Part 1: batch. Memory limits are about 2x the peak seen with docker stats.
    ingest_raw = stage(
        "ingest_raw",
        "uv run python src/ingest_raw.py",
        "2g",
        secrets={"KAGGLE_API_TOKEN": os.environ["KAGGLE_API_TOKEN"]},
    )
    build_bronze = stage("build_bronze", "uv run python src/build_bronze.py", "1536m")
    silver = stage(
        "silver", f"uv run dbt build --select path:models/silver {DBT}", "1g"
    )
    gold = stage("gold", f"uv run dbt build --select path:models/gold {DBT}", "1g")
    train_mlp = stage(
        "train_mlp",
        "uv run python src/train.py mlp --log",
        "2560m",
        environment=THREADS_HEAVY,
        retries=0,
        pool=HEAVY_POOL,
    )
    train_sage = stage(
        "train_sage",
        "uv run python src/train.py sage --log",
        "3g",
        environment=THREADS_HEAVY,
        retries=0,
        pool=HEAVY_POOL,
    )
    write_predictions = stage(
        "write_predictions",
        "uv run python src/write_predictions.py",
        "2g",
        environment={**POSTGRES, **THREADS_HEAVY},
        secrets=POSTGRES_SECRET,
        pool=HEAVY_POOL,
    )

    # Part 2: replay through Kafka, then compare with the batch predictions
    recreate_topic = stage(
        "recreate_topic",
        [
            (
                f"{KAFKA_TOPICS} --delete --topic transactions --if-exists; sleep 3; "
                f"{KAFKA_TOPICS} --create --topic transactions "
                "--partitions 3 --replication-factor 1"
            )
        ],
        "1g",
        project=False,
        image=KAFKA_IMAGE,
        entrypoint=["sh", "-c"],
    )
    replay_consume = stage(
        "replay_consume",
        f"uv run python src/consumer.py --run mlp-seed42 --group {GROUP} --idle 60",
        "1g",
        environment={**POSTGRES, **THREADS_LIGHT},
        secrets=POSTGRES_SECRET,
    )
    replay_produce = stage(
        "replay_produce", "uv run python src/producer.py --delay 0.5", "2560m"
    )
    verify_replay = stage(
        "verify_replay",
        "uv run python src/verify_replay.py --run mlp-seed42",
        "1g",
        environment=POSTGRES,
        secrets=POSTGRES_SECRET,
    )

    # Services the pipeline turns on and off: setup starts, teardown always stops.
    start_postgres = start_service.override(task_id="start_postgres")(
        POSTGRES_CONTAINER, POSTGRES_READY
    )
    stop_postgres = stop_service.override(task_id="stop_postgres")(POSTGRES_CONTAINER)
    start_kafka = start_service.override(task_id="start_kafka")(
        KAFKA_CONTAINER, KAFKA_READY
    )
    stop_kafka = stop_service.override(task_id="stop_kafka")(KAFKA_CONTAINER)
    start_postgres.as_setup()
    start_kafka.as_setup()
    stop_postgres.as_teardown(setups=start_postgres, on_failure_fail_dagrun=True)
    stop_kafka.as_teardown(setups=start_kafka, on_failure_fail_dagrun=True)

    ingest_raw >> build_bronze >> silver >> gold >> train_mlp >> train_sage
    train_sage >> start_postgres >> write_predictions >> start_kafka >> recreate_topic
    recreate_topic >> [replay_consume, replay_produce] >> verify_replay
    verify_replay >> [stop_kafka, stop_postgres]
