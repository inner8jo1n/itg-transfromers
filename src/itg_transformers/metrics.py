from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score


def classification_scores(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
