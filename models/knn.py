#!/usr/bin/env python3
"""Hand-written KNN classifier for the seven-class Dry Bean task.

The implementation intentionally does not call machine-learning packages.
It uses the selected standardized features produced by data.select_features.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_DIR = Path("DryBeanDataset") / "selected_features" / "standardized"
DEFAULT_OUTPUT_DIR = Path("DryBeanDataset") / "knn_results"

SPLIT_FILES = {
    "train": "Dry_Bean_Dataset_Selected_Standardized_train.csv",
    "val": "Dry_Bean_Dataset_Selected_Standardized_val.csv",
    "test": "Dry_Bean_Dataset_Selected_Standardized_test.csv",
}

TARGET_COLUMN = "Class"


def read_dataset(path: Path) -> tuple[list[str], list[tuple[float, ...]], list[str]]:
    """读取 CSV，并拆成特征名、特征矩阵和标签。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or TARGET_COLUMN not in reader.fieldnames:
            raise ValueError(f"{path} must contain a {TARGET_COLUMN} column")

        feature_names = [column for column in reader.fieldnames if column != TARGET_COLUMN]
        features: list[tuple[float, ...]] = []
        labels: list[str] = []
        for row in reader:
            features.append(tuple(float(row[column]) for column in feature_names))
            labels.append(row[TARGET_COLUMN])

    return feature_names, features, labels


def squared_euclidean(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """计算欧氏距离的平方"""
    total = 0.0
    for left_value, right_value in zip(left, right):
        diff = left_value - right_value
        total += diff * diff
    return total


def vote(neighbors: Iterable[tuple[float, str]]) -> str:
    """对最近邻投票；若票数相同，用距离总和更小的类别打破平局。"""
    label_counts: Counter[str] = Counter()
    label_distance_sums: defaultdict[str, float] = defaultdict(float)

    for distance, label in neighbors:
        label_counts[label] += 1
        label_distance_sums[label] += distance

    return max(
        label_counts,
        key=lambda label: (
            label_counts[label],
            -label_distance_sums[label],
            label,
        ),
    )


class KNNClassifier:

    def __init__(self, train_features: list[tuple[float, ...]], train_labels: list[str]):
        if len(train_features) != len(train_labels):
            raise ValueError("Feature and label lengths do not match")
        if not train_features:
            raise ValueError("Training set is empty")
        self.train_features = train_features
        self.train_labels = train_labels

    def nearest_neighbors(self, sample: tuple[float, ...], k: int) -> list[tuple[float, str]]:
        # heapq.nsmallest 只保留前 k 个距离，比完整排序更省时间。
        distances = (
            (squared_euclidean(sample, train_sample), label)
            for train_sample, label in zip(self.train_features, self.train_labels)
        )
        return heapq.nsmallest(k, distances, key=lambda item: item[0])

    def predict_one(self, sample: tuple[float, ...], k: int) -> str:
        return vote(self.nearest_neighbors(sample, k))

    def predict_many(self, samples: list[tuple[float, ...]], k: int) -> list[str]:
        return [self.predict_one(sample, k) for sample in samples]

    def predict_many_for_k_values(
        self,
        samples: list[tuple[float, ...]],
        k_values: list[int],
    ) -> dict[int, list[str]]:
        """一次找出最大 k 个邻居，再复用给多个 k 值，避免重复计算距离。"""
        max_k = max(k_values)
        predictions = {k: [] for k in k_values}

        for sample in samples:
            nearest = self.nearest_neighbors(sample, max_k)
            for k in k_values:
                predictions[k].append(vote(nearest[:k]))

        return predictions


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
    correct = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == predicted)
    return correct / len(y_true)


def confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> list[list[int]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for expected, predicted in zip(y_true, y_pred):
        matrix[label_to_index[expected]][label_to_index[predicted]] += 1

    return matrix


def classification_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """手写 precision / recall / f1，便于后续比较不同模型。"""
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
        tp = matrix[index][index]
        fp = sum(matrix[row_index][index] for row_index in range(len(labels))) - tp
        fn = sum(matrix[index]) - tp
        support = sum(matrix[index])

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
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


def parse_k_values(raw_value: str) -> list[int]:
    values = []
    for part in raw_value.split(","):
        value = int(part.strip())
        if value <= 0:
            raise ValueError("k must be positive")
        values.append(value)
    return sorted(set(values))


def write_predictions(
    path: Path,
    feature_names: list[str],
    features: list[tuple[float, ...]],
    y_true: list[str],
    y_pred: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = feature_names + ["true_class", "predicted_class", "is_correct"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample, expected, predicted in zip(features, y_true, y_pred):
            row = {name: value for name, value in zip(feature_names, sample)}
            row["true_class"] = expected
            row["predicted_class"] = predicted
            row["is_correct"] = int(expected == predicted)
            writer.writerow(row)


def write_confusion_matrix(path: Path, labels: list[str], matrix: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--k-values",
        default="1,3,5,7,9,11,13,15,17,19,21",
        help="Comma-separated candidate k values selected on the validation set.",
    )
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)

    train_path = args.data_dir / SPLIT_FILES["train"]
    val_path = args.data_dir / SPLIT_FILES["val"]
    test_path = args.data_dir / SPLIT_FILES["test"]

    feature_names, train_features, train_labels = read_dataset(train_path)
    val_feature_names, val_features, val_labels = read_dataset(val_path)
    test_feature_names, test_features, test_labels = read_dataset(test_path)

    if feature_names != val_feature_names or feature_names != test_feature_names:
        raise ValueError("Feature columns differ between splits")

    # 只用训练集作为 KNN 的“记忆库”，验证集只负责选 k，测试集只做最终评估。
    classifier = KNNClassifier(train_features, train_labels)
    val_predictions_by_k = classifier.predict_many_for_k_values(val_features, k_values)
    validation_scores = {
        str(k): accuracy_score(val_labels, predictions)
        for k, predictions in val_predictions_by_k.items()
    }

    # 准确率相同则选更小的 k，让模型尽量简单。
    best_k = max(k_values, key=lambda k: (validation_scores[str(k)], -k))
    val_predictions = val_predictions_by_k[best_k]
    test_predictions = classifier.predict_many(test_features, best_k)

    labels = sorted(set(train_labels) | set(val_labels) | set(test_labels))
    val_report = classification_report(val_labels, val_predictions, labels)
    test_report = classification_report(test_labels, test_predictions, labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_predictions_path = args.output_dir / "knn_val_predictions.csv"
    test_predictions_path = args.output_dir / "knn_test_predictions.csv"
    test_confusion_matrix_path = args.output_dir / "knn_test_confusion_matrix.csv"
    report_path = args.output_dir / "knn_report.json"

    write_predictions(
        val_predictions_path,
        feature_names,
        val_features,
        val_labels,
        val_predictions,
    )
    write_predictions(
        test_predictions_path,
        feature_names,
        test_features,
        test_labels,
        test_predictions,
    )
    write_confusion_matrix(
        test_confusion_matrix_path,
        labels,
        test_report["confusion_matrix"],
    )

    report = {
        "algorithm": "hand_written_knn",
        "distance": "squared_euclidean",
        "voting": "majority_vote_with_distance_sum_tie_break",
        "data_dir": str(args.data_dir),
        "feature_count": len(feature_names),
        "features": feature_names,
        "train_size": len(train_features),
        "val_size": len(val_features),
        "test_size": len(test_features),
        "candidate_k_values": k_values,
        "validation_accuracy_by_k": validation_scores,
        "best_k": best_k,
        "validation_report": val_report,
        "test_report": test_report,
        "outputs": {
            "val_predictions": str(val_predictions_path),
            "test_predictions": str(test_predictions_path),
            "test_confusion_matrix": str(test_confusion_matrix_path),
        },
    }

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Hand-written KNN finished")
    print(f"Features:      {len(feature_names)}")
    print(f"Train/val/test:{len(train_features)}/{len(val_features)}/{len(test_features)}")
    print(f"Best k:        {best_k}")
    print(f"Val accuracy:  {val_report['accuracy']:.6f}")
    print(f"Test accuracy: {test_report['accuracy']:.6f}")
    print(f"Test macro F1: {test_report['macro_avg']['f1']:.6f}")
    print(f"Report:        {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
