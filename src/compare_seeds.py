"""Train each model with several seeds, log every run to MLflow and summarize."""

import statistics

from load_gold import load_gold
from metrics import GROUPS, f1_in_steps
from tracking import setup
from train import MODELS, train_logged

SEEDS = [0, 1, 2, 3, 4]


def main():
    data = load_gold()
    setup()
    for name in MODELS:
        scores = {group: [] for group in GROUPS}
        for seed in SEEDS:
            _, pred = train_logged(name, data, seed=seed, verbose=False)
            row = {g: f1_in_steps(pred, data, *span) for g, span in GROUPS.items()}
            for g, value in row.items():
                scores[g].append(value)
            print(name, "seed", seed, {g: round(v, 3) for g, v in row.items()})
        for g, values in scores.items():
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            print(f"{name:5s} {g:6s} f1 mean {mean:.3f}  std {std:.3f}")


if __name__ == "__main__":
    main()
