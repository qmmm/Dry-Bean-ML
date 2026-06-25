#!/usr/bin/env python3
"""Hand-written ID3-style decision tree and random forest for Dry Bean.

This script avoids machine-learning model packages. It implements entropy,
information gain, continuous-threshold splitting, bootstrap sampling, feature
subsampling, majority voting, and evaluation metrics directly in Python.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_DIR = Path("DryBeanDataset") / "selected_features" / "standardized"
DEFAULT_OUTPUT_DIR = Path("DryBeanDataset") / "tree_model_results"

SPLIT_FILES = {
    "train": "Dry_Bean_Dataset_Selected_Standardized_train.csv",
    "val": "Dry_Bean_Dataset_Selected_Standardized_val.csv",
    "test": "Dry_Bean_Dataset_Selected_Standardized_test.csv",
}

TARGET_COLUMN = "Class"


@dataclass
class TreeNode:
    """决策树节点；叶子节点只使用 prediction，内部节点使用 feature/threshold。"""

    prediction: int
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


def read_dataset(path: Path) -> tuple[list[str], list[tuple[float, ...]], list[str]]:
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


def encode_labels(
    train_labels: list[str],
    val_labels: list[str],
    test_labels: list[str],
) -> tuple[list[int], list[int], list[int], list[str]]:
    labels = sorted(set(train_labels) | set(val_labels) | set(test_labels))
    label_to_index = {label: index for index, label in enumerate(labels)}
    return (
        [label_to_index[label] for label in train_labels],
        [label_to_index[label] for label in val_labels],
        [label_to_index[label] for label in test_labels],
        labels,
    )


def entropy_from_counts(counts: list[int]) -> float:
    """计算信息熵：类别越混杂，熵越高。"""
    total = sum(counts)
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts:
        if count:
            probability = count / total
            entropy -= probability * math.log2(probability)
    return entropy


def majority_label(labels: list[int], indices: list[int], n_classes: int) -> int:
    counts = [0] * n_classes
    for index in indices:
        counts[labels[index]] += 1
    max_count = max(counts)
    return counts.index(max_count)


def class_counts(labels: list[int], indices: list[int], n_classes: int) -> list[int]:
    counts = [0] * n_classes
    for index in indices:
        counts[labels[index]] += 1
    return counts


class ID3DecisionTree:
    """支持连续特征的 ID3 风格决策树。

    原始 ID3 多用于离散特征；这里用“二分阈值 + 信息增益”来处理连续特征。
    """

    def __init__(
        self,
        max_depth: int = 12,
        min_samples_split: int = 10,
        min_samples_leaf: int = 4,
        min_gain: float = 1e-7,
        max_features: int | None = None,
        random_state: int | None = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.max_features = max_features
        self.random = random.Random(random_state)
        self.root: TreeNode | None = None
        self.n_features = 0
        self.n_classes = 0
        self.node_count = 0
        self.leaf_count = 0
        self.max_observed_depth = 0

    def fit(
        self,
        features: list[tuple[float, ...]],
        labels: list[int],
        indices: list[int] | None = None,
    ) -> "ID3DecisionTree":
        self.n_features = len(features[0])
        self.n_classes = max(labels) + 1
        train_indices = indices if indices is not None else list(range(len(features)))
        self.node_count = 0
        self.leaf_count = 0
        self.max_observed_depth = 0
        self.root = self._build_node(features, labels, train_indices, depth=0)
        return self

    def _choose_feature_subset(self) -> list[int]:
        all_features = list(range(self.n_features))
        if self.max_features is None or self.max_features >= self.n_features:
            return all_features
        return self.random.sample(all_features, self.max_features)

    def _best_split(
        self,
        features: list[tuple[float, ...]],
        labels: list[int],
        indices: list[int],
        candidate_features: list[int],
    ) -> tuple[int | None, float | None, float]:
        parent_counts = class_counts(labels, indices, self.n_classes)
        parent_entropy = entropy_from_counts(parent_counts)
        best_feature = None
        best_threshold = None
        best_gain = 0.0
        sample_count = len(indices)

        for feature_index in candidate_features:
            sorted_pairs = sorted(
                (features[index][feature_index], labels[index]) for index in indices
            )
            total_counts = parent_counts[:]
            left_counts = [0] * self.n_classes

            for position in range(sample_count - 1):
                value, label = sorted_pairs[position]
                left_counts[label] += 1

                next_value, next_label = sorted_pairs[position + 1]
                if value == next_value:
                    continue
                # 相邻样本标签相同的阈值通常不会成为最优分割点，跳过可减少计算。
                if label == next_label:
                    continue

                left_size = position + 1
                right_size = sample_count - left_size
                if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                    continue

                right_counts = [
                    total_counts[class_index] - left_counts[class_index]
                    for class_index in range(self.n_classes)
                ]
                weighted_entropy = (
                    left_size / sample_count * entropy_from_counts(left_counts)
                    + right_size / sample_count * entropy_from_counts(right_counts)
                )
                gain = parent_entropy - weighted_entropy
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = (value + next_value) / 2.0

        return best_feature, best_threshold, best_gain

    def _build_node(
        self,
        features: list[tuple[float, ...]],
        labels: list[int],
        indices: list[int],
        depth: int,
    ) -> TreeNode:
        prediction = majority_label(labels, indices, self.n_classes)
        node = TreeNode(prediction=prediction, depth=depth)
        self.node_count += 1
        self.max_observed_depth = max(self.max_observed_depth, depth)

        counts = class_counts(labels, indices, self.n_classes)
        is_pure = sum(1 for count in counts if count > 0) == 1
        if (
            is_pure
            or depth >= self.max_depth
            or len(indices) < self.min_samples_split
        ):
            self.leaf_count += 1
            return node

        best_feature, best_threshold, best_gain = self._best_split(
            features,
            labels,
            indices,
            self._choose_feature_subset(),
        )
        if best_feature is None or best_threshold is None or best_gain < self.min_gain:
            self.leaf_count += 1
            return node

        left_indices = []
        right_indices = []
        for index in indices:
            if features[index][best_feature] <= best_threshold:
                left_indices.append(index)
            else:
                right_indices.append(index)

        if (
            len(left_indices) < self.min_samples_leaf
            or len(right_indices) < self.min_samples_leaf
        ):
            self.leaf_count += 1
            return node

        node.feature_index = best_feature
        node.threshold = best_threshold
        node.left = self._build_node(features, labels, left_indices, depth + 1)
        node.right = self._build_node(features, labels, right_indices, depth + 1)
        return node

    def predict_one(self, sample: tuple[float, ...]) -> int:
        if self.root is None:
            raise ValueError("Tree has not been fitted")

        node = self.root
        while not node.is_leaf:
            assert node.feature_index is not None
            assert node.threshold is not None
            if sample[node.feature_index] <= node.threshold:
                assert node.left is not None
                node = node.left
            else:
                assert node.right is not None
                node = node.right
        return node.prediction

    def predict_many(self, samples: list[tuple[float, ...]]) -> list[int]:
        return [self.predict_one(sample) for sample in samples]


class RandomForestClassifierFromScratch:
    """手写随机森林：多棵随机化 ID3 树 + 多数投票。"""

    def __init__(
        self,
        n_estimators: int = 41,
        max_depth: int = 14,
        min_samples_split: int = 8,
        min_samples_leaf: int = 3,
        max_features: int | str = "sqrt",
        bootstrap_ratio: float = 0.8,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap_ratio = bootstrap_ratio
        self.random_state = random_state
        self.trees: list[ID3DecisionTree] = []
        self.n_classes = 0
        self.n_features = 0

    def _max_features_count(self) -> int:
        if isinstance(self.max_features, int):
            return max(1, min(self.max_features, self.n_features))
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(self.n_features)))
        if self.max_features == "log2":
            return max(1, int(math.log2(self.n_features)))
        raise ValueError(f"Unsupported max_features: {self.max_features}")

    def fit(
        self,
        features: list[tuple[float, ...]],
        labels: list[int],
    ) -> "RandomForestClassifierFromScratch":
        self.n_features = len(features[0])
        self.n_classes = max(labels) + 1
        sample_size = max(1, int(len(features) * self.bootstrap_ratio))
        max_features_count = self._max_features_count()
        rng = random.Random(self.random_state)
        self.trees = []

        for tree_index in range(self.n_estimators):
            # bootstrap：有放回抽样，让每棵树看到略有不同的数据。
            bootstrap_indices = [rng.randrange(len(features)) for _ in range(sample_size)]
            tree = ID3DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features_count,
                random_state=self.random_state + tree_index + 1,
            )
            tree.fit(features, labels, bootstrap_indices)
            self.trees.append(tree)

        return self

    def predict_one(self, sample: tuple[float, ...]) -> int:
        votes = Counter(tree.predict_one(sample) for tree in self.trees)
        # 平票时选择类别编号小的类别，保证结果可复现。
        return max(votes, key=lambda label: (votes[label], -label))

    def predict_many(self, samples: list[tuple[float, ...]]) -> list[int]:
        return [self.predict_one(sample) for sample in samples]

    def summary(self) -> dict:
        depths = [tree.max_observed_depth for tree in self.trees]
        nodes = [tree.node_count for tree in self.trees]
        leaves = [tree.leaf_count for tree in self.trees]
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap_ratio": self.bootstrap_ratio,
            "observed_depth_min": min(depths),
            "observed_depth_max": max(depths),
            "observed_depth_mean": sum(depths) / len(depths),
            "node_count_mean": sum(nodes) / len(nodes),
            "leaf_count_mean": sum(leaves) / len(leaves),
        }


def accuracy_score(y_true: list[int], y_pred: list[int]) -> float:
    correct = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == predicted)
    return correct / len(y_true)


def confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    label_count: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(label_count)] for _ in range(label_count)]
    for expected, predicted in zip(y_true, y_pred):
        matrix[expected][predicted] += 1
    return matrix


def classification_report(
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> dict:
    matrix = confusion_matrix(y_true, y_pred, len(label_names))
    support_total = len(y_true)
    per_class = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0

    for index, label in enumerate(label_names):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row_index][index] for row_index in range(len(label_names))) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        support = sum(matrix[index])

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

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

    label_count = len(label_names)
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
        "confusion_matrix": matrix,
    }


def decode_predictions(predictions: Iterable[int], label_names: list[str]) -> list[str]:
    return [label_names[prediction] for prediction in predictions]


def write_predictions(
    path: Path,
    feature_names: list[str],
    features: list[tuple[float, ...]],
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = feature_names + ["true_class", "predicted_class", "is_correct"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        true_labels = decode_predictions(y_true, label_names)
        predicted_labels = decode_predictions(y_pred, label_names)
        for sample, expected, predicted in zip(features, true_labels, predicted_labels):
            row = {name: value for name, value in zip(feature_names, sample)}
            row["true_class"] = expected
            row["predicted_class"] = predicted
            row["is_correct"] = int(expected == predicted)
            writer.writerow(row)


def write_confusion_matrix(
    path: Path,
    label_names: list[str],
    matrix: list[list[int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted"] + label_names)
        for label, row in zip(label_names, matrix):
            writer.writerow([label] + row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--id3-max-depth", type=int, default=12)
    parser.add_argument("--id3-min-samples-leaf", type=int, default=4)
    parser.add_argument("--rf-trees", type=int, default=41)
    parser.add_argument("--rf-max-depth", type=int, default=14)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=3)
    parser.add_argument("--rf-bootstrap-ratio", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_path = args.data_dir / SPLIT_FILES["train"]
    val_path = args.data_dir / SPLIT_FILES["val"]
    test_path = args.data_dir / SPLIT_FILES["test"]

    feature_names, train_features, train_label_names = read_dataset(train_path)
    val_feature_names, val_features, val_label_names = read_dataset(val_path)
    test_feature_names, test_features, test_label_names = read_dataset(test_path)
    if feature_names != val_feature_names or feature_names != test_feature_names:
        raise ValueError("Feature columns differ between splits")

    train_labels, val_labels, test_labels, label_names = encode_labels(
        train_label_names,
        val_label_names,
        test_label_names,
    )

    id3 = ID3DecisionTree(
        max_depth=args.id3_max_depth,
        min_samples_leaf=args.id3_min_samples_leaf,
        min_samples_split=args.id3_min_samples_leaf * 2,
        random_state=args.random_state,
    )
    id3.fit(train_features, train_labels)
    id3_val_predictions = id3.predict_many(val_features)
    id3_test_predictions = id3.predict_many(test_features)
    id3_val_report = classification_report(val_labels, id3_val_predictions, label_names)
    id3_test_report = classification_report(test_labels, id3_test_predictions, label_names)

    random_forest = RandomForestClassifierFromScratch(
        n_estimators=args.rf_trees,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        min_samples_split=args.rf_min_samples_leaf * 2,
        bootstrap_ratio=args.rf_bootstrap_ratio,
        random_state=args.random_state,
    )
    random_forest.fit(train_features, train_labels)
    rf_val_predictions = random_forest.predict_many(val_features)
    rf_test_predictions = random_forest.predict_many(test_features)
    rf_val_report = classification_report(val_labels, rf_val_predictions, label_names)
    rf_test_report = classification_report(test_labels, rf_test_predictions, label_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        "id3": {
            "algorithm": "ID3-style entropy/information-gain decision tree",
            "parameters": {
                "max_depth": args.id3_max_depth,
                "min_samples_leaf": args.id3_min_samples_leaf,
                "min_samples_split": args.id3_min_samples_leaf * 2,
            },
            "tree_summary": {
                "node_count": id3.node_count,
                "leaf_count": id3.leaf_count,
                "max_observed_depth": id3.max_observed_depth,
            },
            "validation_report": id3_val_report,
            "test_report": id3_test_report,
        },
        "random_forest": {
            "algorithm": "Random forest from scratch using ID3-style trees",
            "parameters": random_forest.summary(),
            "validation_report": rf_val_report,
            "test_report": rf_test_report,
        },
    }

    comparison_rows = [
        {
            "model": "ID3",
            "val_accuracy": id3_val_report["accuracy"],
            "val_macro_f1": id3_val_report["macro_avg"]["f1"],
            "test_accuracy": id3_test_report["accuracy"],
            "test_macro_f1": id3_test_report["macro_avg"]["f1"],
        },
        {
            "model": "RandomForest",
            "val_accuracy": rf_val_report["accuracy"],
            "val_macro_f1": rf_val_report["macro_avg"]["f1"],
            "test_accuracy": rf_test_report["accuracy"],
            "test_macro_f1": rf_test_report["macro_avg"]["f1"],
        },
    ]

    outputs = {
        "id3_val_predictions": args.output_dir / "id3_val_predictions.csv",
        "id3_test_predictions": args.output_dir / "id3_test_predictions.csv",
        "id3_test_confusion_matrix": args.output_dir / "id3_test_confusion_matrix.csv",
        "rf_val_predictions": args.output_dir / "random_forest_val_predictions.csv",
        "rf_test_predictions": args.output_dir / "random_forest_test_predictions.csv",
        "rf_test_confusion_matrix": args.output_dir / "random_forest_test_confusion_matrix.csv",
        "comparison": args.output_dir / "tree_model_comparison.csv",
        "report": args.output_dir / "tree_model_report.json",
    }

    write_predictions(
        outputs["id3_val_predictions"],
        feature_names,
        val_features,
        val_labels,
        id3_val_predictions,
        label_names,
    )
    write_predictions(
        outputs["id3_test_predictions"],
        feature_names,
        test_features,
        test_labels,
        id3_test_predictions,
        label_names,
    )
    write_confusion_matrix(
        outputs["id3_test_confusion_matrix"],
        label_names,
        id3_test_report["confusion_matrix"],
    )
    write_predictions(
        outputs["rf_val_predictions"],
        feature_names,
        val_features,
        val_labels,
        rf_val_predictions,
        label_names,
    )
    write_predictions(
        outputs["rf_test_predictions"],
        feature_names,
        test_features,
        test_labels,
        rf_test_predictions,
        label_names,
    )
    write_confusion_matrix(
        outputs["rf_test_confusion_matrix"],
        label_names,
        rf_test_report["confusion_matrix"],
    )

    with outputs["comparison"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    full_report = {
        "data_dir": str(args.data_dir),
        "feature_count": len(feature_names),
        "features": feature_names,
        "labels": label_names,
        "train_size": len(train_features),
        "val_size": len(val_features),
        "test_size": len(test_features),
        "models": reports,
        "comparison": comparison_rows,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    with outputs["report"].open("w", encoding="utf-8") as handle:
        json.dump(full_report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Tree models finished")
    print(f"Features:              {len(feature_names)}")
    print(f"Train/val/test:        {len(train_features)}/{len(val_features)}/{len(test_features)}")
    print(
        "ID3 val/test acc:      "
        f"{id3_val_report['accuracy']:.6f}/{id3_test_report['accuracy']:.6f}"
    )
    print(
        "RF val/test acc:       "
        f"{rf_val_report['accuracy']:.6f}/{rf_test_report['accuracy']:.6f}"
    )
    print(
        "ID3/RF test macro F1:  "
        f"{id3_test_report['macro_avg']['f1']:.6f}/{rf_test_report['macro_avg']['f1']:.6f}"
    )
    print(f"Report:                {outputs['report']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
