import numpy as np


def confusion_matrix(y_true, y_pred):

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    # Calculate confusion matrix values
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([
        [tn, fp],
        [fn, tp]
    ])


def evaluate_model(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precision
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0
    )

    # Recall
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0
    )

    # F1-score
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }