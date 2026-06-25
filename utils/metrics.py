"""通用分类指标。"""

from __future__ import annotations


def accuracy_score(y_true: list, y_pred: list) -> float:
    correct = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == predicted)
    return correct / len(y_true)


def confusion_matrix(y_true: list, y_pred: list, labels: list) -> list[list[int]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for expected, predicted in zip(y_true, y_pred):
        matrix[label_to_index[expected]][label_to_index[predicted]] += 1
    return matrix


def classification_report(y_true: list, y_pred: list, labels: list) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels)
    support_total = len(y_true)
    per_class = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(labels))) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        support = sum(matrix[index])
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    label_count = len(labels)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_avg": {
            "precision": macro_precision / label_count,
            "recall": macro_recall / label_count,
            "f1": macro_f1 / label_count,
        },
        "weighted_avg": {
            "precision": weighted_precision / support_total,
            "recall": weighted_recall / support_total,
            "f1": weighted_f1 / support_total,
        },
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": matrix,
    }
