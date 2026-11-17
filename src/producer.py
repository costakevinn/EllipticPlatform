import argparse
import json
import os
import time

from confluent_kafka import Producer

from load_gold import load_gold

TOPIC = "transactions"
BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
FIRST_TEST_STEP = 35
LAST_TEST_STEP = 49


def make_callback(errors):
    def on_delivery(err, msg):
        if err is not None:
            errors.append(str(err))

    return on_delivery


def publish_step(producer, data, step, callback):
    rows = (data["time_step"] == step).nonzero().flatten().tolist()
    for i in rows:
        value = json.dumps(
            {
                "tx_id": int(data["tx_id"][i]),
                "time_step": step,
                "features": data["x"][i].tolist(),
            }
        ).encode()
        while True:
            try:
                producer.produce(
                    TOPIC, key=str(step), value=value, on_delivery=callback
                )
                break
            except BufferError:
                producer.poll(0.5)
        producer.poll(0)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Replay Gold test steps into Kafka")
    parser.add_argument(
        "--delay", type=float, default=2.0, help="seconds between steps"
    )
    parser.add_argument("--first", type=int, default=FIRST_TEST_STEP)
    parser.add_argument("--last", type=int, default=LAST_TEST_STEP)
    args = parser.parse_args()
    if not FIRST_TEST_STEP <= args.first <= args.last <= LAST_TEST_STEP:
        parser.error(
            f"steps must satisfy {FIRST_TEST_STEP} <= first <= last <= {LAST_TEST_STEP}"
        )

    data = load_gold()
    producer = Producer({"bootstrap.servers": BOOTSTRAP, "enable.idempotence": True})
    errors = []
    callback = make_callback(errors)
    total = 0
    for step in range(args.first, args.last + 1):
        count = publish_step(producer, data, step, callback)
        producer.flush()
        if errors:
            raise RuntimeError(f"delivery failed at step {step}: {errors[0]}")
        total += count
        print(f"step {step}: {count} messages")
        if step < args.last:
            time.sleep(args.delay)
    print("total", total)


if __name__ == "__main__":
    main()
