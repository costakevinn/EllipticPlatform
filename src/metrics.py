"""Metrics for the illicit class (label 1)."""


def precision_recall_f1(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def by_time_step(y_true, y_pred, time_step):
    rows = []
    for step in sorted(time_step.unique().tolist()):
        in_step = time_step == step
        p, r, f1 = precision_recall_f1(y_true[in_step], y_pred[in_step])
        n_illicit = int((y_true[in_step] == 1).sum())
        rows.append((step, n_illicit, p, r, f1))
    return rows


GROUPS = {"all": (35, 49), "35_42": (35, 42), "43_49": (43, 49)}


def f1_in_steps(pred, data, first, last):
    steps = data["time_step"]
    mask = data["test_mask"] & (steps >= first) & (steps <= last)
    return precision_recall_f1(data["y"][mask], pred[mask])[2]
