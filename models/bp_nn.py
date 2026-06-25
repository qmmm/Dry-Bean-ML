#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_DIR = Path("DryBeanDataset") / "selected_features" / "standardized"
DEFAULT_OUTPUT_DIR = Path("DryBeanDataset") / "bp_nn_results"

SPLIT_FILES = {
    "train": "Dry_Bean_Dataset_Selected_Standardized_train.csv",
    "val": "Dry_Bean_Dataset_Selected_Standardized_val.csv",
    "test": "Dry_Bean_Dataset_Selected_Standardized_test.csv",
}

TARGET_COLUMN = "Class"


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
    label_names = sorted(set(train_labels) | set(val_labels) | set(test_labels))
    label_to_index = {label: index for index, label in enumerate(label_names)}
    return (
        [label_to_index[label] for label in train_labels],
        [label_to_index[label] for label in val_labels],
        [label_to_index[label] for label in test_labels],
        label_names,
    )


def stable_softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def relu(value: float) -> float:
    return value if value > 0.0 else 0.0


def relu_grad(value: float) -> float:
    return 1.0 if value > 0.0 else 0.0


class BPNeuralNetwork:
    """一隐藏层 BP 神经网络：输入层 -> ReLU 隐藏层 -> softmax 输出层。"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.015,
        l2: float = 5e-4,
        dropout_rate: float = 0.15,
        random_state: int = 42,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.l2 = l2
        self.dropout_rate = dropout_rate
        self.random = random.Random(random_state)

        # He 初始化适合 ReLU，能让初始信号不要过快变大或变小。
        hidden_scale = math.sqrt(2.0 / input_size)
        output_scale = math.sqrt(2.0 / hidden_size)
        self.w1 = [
            [self.random.gauss(0.0, hidden_scale) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [
            [self.random.gauss(0.0, output_scale) for _ in range(hidden_size)]
            for _ in range(output_size)
        ]
        self.b2 = [0.0 for _ in range(output_size)]

    def get_state(self) -> dict:
        return {
            "w1": [row[:] for row in self.w1],
            "b1": self.b1[:],
            "w2": [row[:] for row in self.w2],
            "b2": self.b2[:],
        }

    def set_state(self, state: dict) -> None:
        self.w1 = [row[:] for row in state["w1"]]
        self.b1 = state["b1"][:]
        self.w2 = [row[:] for row in state["w2"]]
        self.b2 = state["b2"][:]

    def forward(
        self,
        sample: tuple[float, ...],
        training: bool = False,
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        hidden_linear = []
        hidden_activated = []
        dropout_mask = []
        keep_probability = 1.0 - self.dropout_rate

        for hidden_index in range(self.hidden_size):
            value = self.b1[hidden_index]
            weights = self.w1[hidden_index]
            for feature_index in range(self.input_size):
                value += weights[feature_index] * sample[feature_index]
            hidden_linear.append(value)
            activation = relu(value)

            if training and self.dropout_rate > 0.0:
                if self.random.random() < self.dropout_rate:
                    mask_value = 0.0
                else:
                    mask_value = 1.0 / keep_probability
                activation *= mask_value
                dropout_mask.append(mask_value)
            else:
                dropout_mask.append(1.0)

            hidden_activated.append(activation)

        logits = []
        for output_index in range(self.output_size):
            value = self.b2[output_index]
            weights = self.w2[output_index]
            for hidden_index in range(self.hidden_size):
                value += weights[hidden_index] * hidden_activated[hidden_index]
            logits.append(value)

        probabilities = stable_softmax(logits)
        return hidden_linear, hidden_activated, logits, probabilities, dropout_mask

    def predict_one(self, sample: tuple[float, ...]) -> int:
        probabilities = self.forward(sample)[3]
        best_index = 0
        best_probability = probabilities[0]
        for index, probability in enumerate(probabilities[1:], start=1):
            if probability > best_probability:
                best_index = index
                best_probability = probability
        return best_index

    def predict_many(self, samples: list[tuple[float, ...]]) -> list[int]:
        return [self.predict_one(sample) for sample in samples]

    def train_one_sample(
        self,
        sample: tuple[float, ...],
        label: int,
        sample_weight: float,
    ) -> float:
        hidden_linear, hidden_activated, _, probabilities, dropout_mask = self.forward(
            sample,
            training=True,
        )

        probability_of_true_class = max(probabilities[label], 1e-15)
        loss = -math.log(probability_of_true_class) * sample_weight

        # softmax + 交叉熵的输出层梯度：p - y。
        output_delta = probabilities[:]
        output_delta[label] -= 1.0
        for output_index in range(self.output_size):
            output_delta[output_index] *= sample_weight

        # 先算隐藏层误差信号，再更新 w2，避免更新后的 w2 污染反传。
        hidden_delta = [0.0 for _ in range(self.hidden_size)]
        for hidden_index in range(self.hidden_size):
            backprop_signal = 0.0
            for output_index in range(self.output_size):
                backprop_signal += self.w2[output_index][hidden_index] * output_delta[output_index]
            hidden_delta[hidden_index] = (
                backprop_signal
                * relu_grad(hidden_linear[hidden_index])
                * dropout_mask[hidden_index]
            )

        lr = self.learning_rate
        l2 = self.l2

        for output_index in range(self.output_size):
            delta = output_delta[output_index]
            weights = self.w2[output_index]
            for hidden_index in range(self.hidden_size):
                gradient = delta * hidden_activated[hidden_index] + l2 * weights[hidden_index]
                weights[hidden_index] -= lr * gradient
            self.b2[output_index] -= lr * delta

        for hidden_index in range(self.hidden_size):
            delta = hidden_delta[hidden_index]
            weights = self.w1[hidden_index]
            for feature_index in range(self.input_size):
                gradient = delta * sample[feature_index] + l2 * weights[feature_index]
                weights[feature_index] -= lr * gradient
            self.b1[hidden_index] -= lr * delta

        return loss

    def average_loss(
        self,
        samples: list[tuple[float, ...]],
        labels: list[int],
    ) -> float:
        total_loss = 0.0
        for sample, label in zip(samples, labels):
            probabilities = self.forward(sample)[3]
            total_loss -= math.log(max(probabilities[label], 1e-15))
        return total_loss / len(samples)


def class_weights(labels: list[int], output_size: int) -> list[float]:
    """按类别频次生成权重，让少数类在损失里更有存在感。"""
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for label in range(output_size):
        weights.append(total / (output_size * counts[label]))
    return weights


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


def train_with_early_stopping(
    network: BPNeuralNetwork,
    train_features: list[tuple[float, ...]],
    train_labels: list[int],
    val_features: list[tuple[float, ...]],
    val_labels: list[int],
    epochs: int,
    patience: int,
    random_state: int,
) -> tuple[dict, list[dict]]:
    rng = random.Random(random_state)
    indices = list(range(len(train_features)))
    weights = class_weights(train_labels, network.output_size)
    best_state = network.get_state()
    best_val_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        running_loss = 0.0

        for index in indices:
            label = train_labels[index]
            running_loss += network.train_one_sample(
                train_features[index],
                label,
                weights[label],
            )

        val_predictions = network.predict_many(val_features)
        train_predictions = network.predict_many(train_features)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        val_accuracy = accuracy_score(val_labels, val_predictions)
        train_update_loss = running_loss / len(train_features)
        # 用关闭 dropout 的普通交叉熵记录曲线，和验证集 loss 保持同口径。
        train_loss = network.average_loss(train_features, train_labels)
        val_loss = network.average_loss(val_features, val_labels)
        history_row = {
            "epoch": epoch,
            "train_update_loss": train_update_loss,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "learning_rate": network.learning_rate,
        }
        history.append(history_row)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = network.get_state()
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"train_acc={train_accuracy:.5f} val_acc={val_accuracy:.5f}"
        )

        if stale_epochs >= patience:
            break

        # 简单学习率衰减：越到后面步子越小，帮助收敛。
        if epoch % 10 == 0:
            network.learning_rate *= 0.85

    network.set_state(best_state)
    return {"best_epoch": best_epoch, "best_val_accuracy": best_val_accuracy}, history


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.015)
    parser.add_argument("--l2", type=float, default=5e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.dropout_rate < 1.0:
        raise ValueError("--dropout-rate must be in [0, 1)")

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

    network = BPNeuralNetwork(
        input_size=len(feature_names),
        hidden_size=args.hidden_size,
        output_size=len(label_names),
        learning_rate=args.learning_rate,
        l2=args.l2,
        dropout_rate=args.dropout_rate,
        random_state=args.random_state,
    )
    best_info, history = train_with_early_stopping(
        network,
        train_features,
        train_labels,
        val_features,
        val_labels,
        epochs=args.epochs,
        patience=args.patience,
        random_state=args.random_state,
    )

    val_predictions = network.predict_many(val_features)
    test_predictions = network.predict_many(test_features)
    val_report = classification_report(val_labels, val_predictions, label_names)
    test_report = classification_report(test_labels, test_predictions, label_names)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_predictions_path = args.output_dir / "bp_nn_val_predictions.csv"
    test_predictions_path = args.output_dir / "bp_nn_test_predictions.csv"
    test_confusion_matrix_path = args.output_dir / "bp_nn_test_confusion_matrix.csv"
    history_path = args.output_dir / "bp_nn_training_history.csv"
    report_path = args.output_dir / "bp_nn_report.json"

    write_predictions(
        val_predictions_path,
        feature_names,
        val_features,
        val_labels,
        val_predictions,
        label_names,
    )
    write_predictions(
        test_predictions_path,
        feature_names,
        test_features,
        test_labels,
        test_predictions,
        label_names,
    )
    write_confusion_matrix(
        test_confusion_matrix_path,
        label_names,
        test_report["confusion_matrix"],
    )
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    report = {
        "algorithm": "hand_written_bp_neural_network",
        "architecture": {
            "input_size": len(feature_names),
            "hidden_size": args.hidden_size,
            "output_size": len(label_names),
            "hidden_activation": "relu",
            "output_activation": "softmax",
            "loss": "weighted_cross_entropy",
            "optimizer": "stochastic_gradient_descent",
            "regularization": {
                "l2": args.l2,
                "dropout_rate": args.dropout_rate,
                "dropout_scope": "hidden layer during training only",
            },
        },
        "data_dir": str(args.data_dir),
        "features": feature_names,
        "labels": label_names,
        "train_size": len(train_features),
        "val_size": len(val_features),
        "test_size": len(test_features),
        "parameters": {
            "epochs_requested": args.epochs,
            "epochs_ran": len(history),
            "learning_rate_initial": args.learning_rate,
            "learning_rate_final": network.learning_rate,
            "l2": args.l2,
            "dropout_rate": args.dropout_rate,
            "patience": args.patience,
            "random_state": args.random_state,
        },
        "best_validation": best_info,
        "validation_report": val_report,
        "test_report": test_report,
        "outputs": {
            "val_predictions": str(val_predictions_path),
            "test_predictions": str(test_predictions_path),
            "test_confusion_matrix": str(test_confusion_matrix_path),
            "training_history": str(history_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("BP neural network finished")
    print(f"Features:       {len(feature_names)}")
    print(f"Hidden size:    {args.hidden_size}")
    print(f"Best epoch:     {best_info['best_epoch']}")
    print(f"Val accuracy:   {val_report['accuracy']:.6f}")
    print(f"Test accuracy:  {test_report['accuracy']:.6f}")
    print(f"Test macro F1:  {test_report['macro_avg']['f1']:.6f}")
    print(f"Report:         {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
