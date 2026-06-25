#!/usr/bin/env python3
"""对 Dry Bean 分类算法进行过拟合实验和绘图分析。"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from models.bp_nn import (
    BPNeuralNetwork,
    accuracy_score as bp_accuracy_score,
    class_weights,
    encode_labels as encode_labels_for_bp,
)
from models.tree import (
    ID3DecisionTree,
    RandomForestClassifierFromScratch,
    accuracy_score,
    encode_labels,
    read_dataset,
)


DEFAULT_DATA_DIR = Path("DryBeanDataset") / "selected_features" / "standardized"
DEFAULT_OUTPUT_DIR = Path("DryBeanDataset") / "overfitting_analysis_results"

SPLIT_FILES = {
    "train": "Dry_Bean_Dataset_Selected_Standardized_train.csv",
    "val": "Dry_Bean_Dataset_Selected_Standardized_val.csv",
    "test": "Dry_Bean_Dataset_Selected_Standardized_test.csv",
}


def parse_int_list(raw_value: str) -> list[int]:
    values = []
    for part in raw_value.split(","):
        stripped = part.strip()
        if stripped:
            value = int(stripped)
            if value <= 0:
                raise ValueError("integer list values must be positive")
            values.append(value)
    return values


def accuracy_from_label_names(y_true: list[str], y_pred: np.ndarray, label_names: list[str]) -> float:
    label_to_index = {label: index for index, label in enumerate(label_names)}
    expected = np.asarray([label_to_index[label] for label in y_true], dtype=np.int64)
    return float(np.mean(expected == y_pred))


def predict_knn_numpy(
    train_array: np.ndarray,
    query_array: np.ndarray,
    train_label_indices: np.ndarray,
    label_names: list[str],
    k: int,
    batch_size: int,
) -> np.ndarray:
    predictions = np.empty(len(query_array), dtype=np.int64)
    train_norms = np.sum(train_array * train_array, axis=1)
    label_count = len(label_names)

    for start in range(0, len(query_array), batch_size):
        stop = min(start + batch_size, len(query_array))
        batch = query_array[start:stop]
        distances = (
            np.sum(batch * batch, axis=1, keepdims=True)
            + train_norms
            - 2.0 * batch @ train_array.T
        )
        nearest_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]

        for row_index, neighbors in enumerate(nearest_indices):
            counts = np.zeros(label_count, dtype=np.int64)
            distance_sums = np.zeros(label_count, dtype=np.float64)
            row_distances = distances[row_index]
            for neighbor in neighbors:
                label_index = int(train_label_indices[neighbor])
                counts[label_index] += 1
                distance_sums[label_index] += row_distances[neighbor]

            best_label = 0
            for label_index in range(1, label_count):
                current_key = (
                    counts[label_index],
                    -distance_sums[label_index],
                    label_names[label_index],
                )
                best_key = (
                    counts[best_label],
                    -distance_sums[best_label],
                    label_names[best_label],
                )
                if current_key > best_key:
                    best_label = label_index
            predictions[start + row_index] = best_label

    return predictions


def predict_knn_many_numpy(
    train_array: np.ndarray,
    query_array: np.ndarray,
    train_label_indices: np.ndarray,
    label_names: list[str],
    k_values: list[int],
    batch_size: int,
) -> dict[int, np.ndarray]:
    max_k = max(k_values)
    predictions = {k: np.empty(len(query_array), dtype=np.int64) for k in k_values}
    train_norms = np.sum(train_array * train_array, axis=1)
    label_count = len(label_names)

    for start in range(0, len(query_array), batch_size):
        stop = min(start + batch_size, len(query_array))
        batch = query_array[start:stop]
        distances = (
            np.sum(batch * batch, axis=1, keepdims=True)
            + train_norms
            - 2.0 * batch @ train_array.T
        )
        nearest_indices = np.argpartition(distances, kth=max_k - 1, axis=1)[:, :max_k]

        for row_index, neighbors in enumerate(nearest_indices):
            row_distances = distances[row_index]
            sorted_neighbors = neighbors[np.argsort(row_distances[neighbors])]
            counts = np.zeros(label_count, dtype=np.int64)
            distance_sums = np.zeros(label_count, dtype=np.float64)
            k_position = 0

            for rank, neighbor in enumerate(sorted_neighbors, start=1):
                label_index = int(train_label_indices[neighbor])
                counts[label_index] += 1
                distance_sums[label_index] += row_distances[neighbor]

                if k_position < len(k_values) and rank == k_values[k_position]:
                    best_label = 0
                    for candidate in range(1, label_count):
                        current_key = (
                            counts[candidate],
                            -distance_sums[candidate],
                            label_names[candidate],
                        )
                        best_key = (
                            counts[best_label],
                            -distance_sums[best_label],
                            label_names[best_label],
                        )
                        if current_key > best_key:
                            best_label = candidate
                    predictions[k_values[k_position]][start + row_index] = best_label
                    k_position += 1

    return predictions


def add_result(
    rows: list[dict],
    experiment: str,
    model: str,
    parameter_name: str,
    parameter_value: int,
    train_accuracy: float,
    val_accuracy: float,
    test_accuracy: float,
    extra: dict | None = None,
) -> None:
    row = {
        "experiment": experiment,
        "model": model,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "train_val_gap": train_accuracy - val_accuracy,
        "train_test_gap": train_accuracy - test_accuracy,
    }
    if extra:
        row.update(extra)
    rows.append(row)


def run_knn_experiment(
    rows: list[dict],
    train_features: list[tuple[float, ...]],
    train_labels: list[str],
    val_features: list[tuple[float, ...]],
    val_labels: list[str],
    test_features: list[tuple[float, ...]],
    test_labels: list[str],
    k_values: list[int],
    batch_size: int,
) -> None:
    print("Running KNN k-value overfitting experiment...")
    label_names = sorted(set(train_labels) | set(val_labels) | set(test_labels))
    label_to_index = {label: index for index, label in enumerate(label_names)}
    train_array = np.asarray(train_features, dtype=np.float64)
    val_array = np.asarray(val_features, dtype=np.float64)
    test_array = np.asarray(test_features, dtype=np.float64)
    train_label_indices = np.asarray([label_to_index[label] for label in train_labels], dtype=np.int64)

    k_values = sorted(k_values)
    train_predictions = predict_knn_many_numpy(
        train_array,
        train_array,
        train_label_indices,
        label_names,
        k_values,
        batch_size,
    )
    val_predictions = predict_knn_many_numpy(
        train_array,
        val_array,
        train_label_indices,
        label_names,
        k_values,
        batch_size,
    )
    test_predictions = predict_knn_many_numpy(
        train_array,
        test_array,
        train_label_indices,
        label_names,
        k_values,
        batch_size,
    )

    for k in k_values:
        train_pred = train_predictions[k]
        val_pred = val_predictions[k]
        test_pred = test_predictions[k]
        add_result(
            rows,
            "knn_k_curve",
            "KNN",
            "k",
            k,
            accuracy_from_label_names(train_labels, train_pred, label_names),
            accuracy_from_label_names(val_labels, val_pred, label_names),
            accuracy_from_label_names(test_labels, test_pred, label_names),
        )


def run_tree_depth_experiment(
    rows: list[dict],
    train_features: list[tuple[float, ...]],
    train_labels: list[int],
    val_features: list[tuple[float, ...]],
    val_labels: list[int],
    test_features: list[tuple[float, ...]],
    test_labels: list[int],
    depths: list[int],
    random_state: int,
) -> None:
    print("Running ID3 depth overfitting experiment...")
    for depth in depths:
        tree = ID3DecisionTree(
            max_depth=depth,
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=random_state,
        )
        tree.fit(train_features, train_labels)
        add_result(
            rows,
            "id3_depth_curve",
            "ID3",
            "max_depth",
            depth,
            accuracy_score(train_labels, tree.predict_many(train_features)),
            accuracy_score(val_labels, tree.predict_many(val_features)),
            accuracy_score(test_labels, tree.predict_many(test_features)),
            {
                "node_count": tree.node_count,
                "leaf_count": tree.leaf_count,
                "observed_depth": tree.max_observed_depth,
            },
        )


def run_random_forest_depth_experiment(
    rows: list[dict],
    train_features: list[tuple[float, ...]],
    train_labels: list[int],
    val_features: list[tuple[float, ...]],
    val_labels: list[int],
    test_features: list[tuple[float, ...]],
    test_labels: list[int],
    depths: list[int],
    n_estimators: int,
    random_state: int,
) -> None:
    print("Running Random Forest depth overfitting experiment...")
    for depth in depths:
        forest = RandomForestClassifierFromScratch(
            n_estimators=n_estimators,
            max_depth=depth,
            min_samples_leaf=3,
            min_samples_split=6,
            bootstrap_ratio=0.8,
            random_state=random_state,
        )
        forest.fit(train_features, train_labels)
        summary = forest.summary()
        add_result(
            rows,
            "random_forest_depth_curve",
            "Random Forest",
            "max_depth",
            depth,
            accuracy_score(train_labels, forest.predict_many(train_features)),
            accuracy_score(val_labels, forest.predict_many(val_features)),
            accuracy_score(test_labels, forest.predict_many(test_features)),
            {
                "n_estimators": n_estimators,
                "observed_depth_mean": summary["observed_depth_mean"],
                "leaf_count_mean": summary["leaf_count_mean"],
            },
        )


def run_bp_epoch_experiment(
    rows: list[dict],
    train_features: list[tuple[float, ...]],
    train_labels: list[int],
    val_features: list[tuple[float, ...]],
    val_labels: list[int],
    test_features: list[tuple[float, ...]],
    test_labels: list[int],
    input_size: int,
    output_size: int,
    epochs: int,
    hidden_size: int,
    learning_rate: float,
    l2: float,
    dropout_rate: float,
    random_state: int,
) -> None:
    print("Running BP epoch overfitting experiment...")
    network = BPNeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate,
        l2=l2,
        dropout_rate=dropout_rate,
        random_state=random_state,
    )
    rng = random.Random(random_state)
    indices = list(range(len(train_features)))
    weights = class_weights(train_labels, output_size)

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        with contextlib.redirect_stdout(io.StringIO()):
            for index in indices:
                label = train_labels[index]
                network.train_one_sample(train_features[index], label, weights[label])
        if epoch % 10 == 0:
            network.learning_rate *= 0.85

        train_predictions = network.predict_many(train_features)
        val_predictions = network.predict_many(val_features)
        test_predictions = network.predict_many(test_features)
        add_result(
            rows,
            "bp_epoch_curve",
            "BP Neural Network",
            "epoch",
            epoch,
            bp_accuracy_score(train_labels, train_predictions),
            bp_accuracy_score(val_labels, val_predictions),
            bp_accuracy_score(test_labels, test_predictions),
            {
                "train_loss": network.average_loss(train_features, train_labels),
                "val_loss": network.average_loss(val_features, val_labels),
                "learning_rate": network.learning_rate,
            },
        )


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "experiment",
        "model",
        "parameter_name",
        "parameter_value",
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "train_val_gap",
        "train_test_gap",
    ]
    fieldnames = preferred + [field for field in fieldnames if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_accuracy_curves(rows: list[dict], output_path: Path) -> None:
    experiments = [
        ("knn_k_curve", "KNN: k vs overfitting"),
        ("id3_depth_curve", "ID3: max depth vs overfitting"),
        ("random_forest_depth_curve", "Random Forest: max depth vs overfitting"),
        ("bp_epoch_curve", "BP: epoch vs overfitting"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {
        "train_accuracy": "#4e79a7",
        "val_accuracy": "#f28e2b",
        "test_accuracy": "#59a14f",
    }

    for axis, (experiment, title) in zip(axes.flatten(), experiments):
        subset = sorted(
            [row for row in rows if row["experiment"] == experiment],
            key=lambda row: row["parameter_value"],
        )
        x_values = [row["parameter_value"] for row in subset]
        for metric, label in [
            ("train_accuracy", "Train"),
            ("val_accuracy", "Validation"),
            ("test_accuracy", "Test"),
        ]:
            axis.plot(
                x_values,
                [row[metric] for row in subset],
                marker="o",
                linewidth=1.7,
                color=colors[metric],
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel(subset[0]["parameter_name"] if subset else "")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0.65, 1.01)
        axis.grid(True, alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Overfitting Analysis: Train/Validation/Test Accuracy", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_gap_curves(rows: list[dict], output_path: Path) -> None:
    experiments = [
        ("knn_k_curve", "KNN"),
        ("id3_depth_curve", "ID3"),
        ("random_forest_depth_curve", "Random Forest"),
        ("bp_epoch_curve", "BP Neural Network"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for axis, (experiment, title) in zip(axes.flatten(), experiments):
        subset = sorted(
            [row for row in rows if row["experiment"] == experiment],
            key=lambda row: row["parameter_value"],
        )
        x_values = [row["parameter_value"] for row in subset]
        axis.plot(
            x_values,
            [row["train_val_gap"] for row in subset],
            marker="o",
            linewidth=1.7,
            label="Train - Validation",
            color="#e15759",
        )
        axis.plot(
            x_values,
            [row["train_test_gap"] for row in subset],
            marker="s",
            linewidth=1.7,
            label="Train - Test",
            color="#7f7f7f",
        )
        axis.axhline(0.0, color="#333333", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel(subset[0]["parameter_name"] if subset else "")
        axis.set_ylabel("Accuracy gap")
        axis.grid(True, alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Overfitting Analysis: Generalization Gap", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def best_rows(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for experiment in sorted({row["experiment"] for row in rows}):
        subset = [row for row in rows if row["experiment"] == experiment]
        best_val = max(subset, key=lambda row: (row["val_accuracy"], row["test_accuracy"]))
        largest_gap = max(subset, key=lambda row: row["train_test_gap"])
        result[experiment] = {
            "best_validation": best_val,
            "largest_train_test_gap": largest_gap,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-values", default="1,3,5,7,11,15,21,31")
    parser.add_argument("--tree-depths", default="2,4,6,8,10,12,16,20")
    parser.add_argument("--rf-depths", default="2,4,6,8,10,12,16")
    parser.add_argument("--rf-trees", type=int, default=9)
    parser.add_argument("--bp-epochs", type=int, default=30)
    parser.add_argument("--bp-hidden-size", type=int, default=32)
    parser.add_argument("--bp-learning-rate", type=float, default=0.015)
    parser.add_argument("--bp-l2", type=float, default=5e-4)
    parser.add_argument("--bp-dropout-rate", type=float, default=0.15)
    parser.add_argument("--knn-batch-size", type=int, default=128)
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
    bp_train_labels, bp_val_labels, bp_test_labels, bp_label_names = encode_labels_for_bp(
        train_label_names,
        val_label_names,
        test_label_names,
    )
    if label_names != bp_label_names:
        raise ValueError("Label encoders differ between models")

    rows: list[dict] = []
    run_knn_experiment(
        rows,
        train_features,
        train_label_names,
        val_features,
        val_label_names,
        test_features,
        test_label_names,
        parse_int_list(args.k_values),
        args.knn_batch_size,
    )
    run_tree_depth_experiment(
        rows,
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        parse_int_list(args.tree_depths),
        args.random_state,
    )
    run_random_forest_depth_experiment(
        rows,
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        parse_int_list(args.rf_depths),
        args.rf_trees,
        args.random_state,
    )
    run_bp_epoch_experiment(
        rows,
        train_features,
        bp_train_labels,
        val_features,
        bp_val_labels,
        test_features,
        bp_test_labels,
        len(feature_names),
        len(label_names),
        args.bp_epochs,
        args.bp_hidden_size,
        args.bp_learning_rate,
        args.bp_l2,
        args.bp_dropout_rate,
        args.random_state,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "overfitting_summary.csv"
    report_path = args.output_dir / "overfitting_report.json"
    accuracy_plot_path = args.output_dir / "overfitting_accuracy_curves.png"
    gap_plot_path = args.output_dir / "overfitting_gap_curves.png"

    write_summary_csv(summary_path, rows)
    plot_accuracy_curves(rows, accuracy_plot_path)
    plot_gap_curves(rows, gap_plot_path)

    report = {
        "data_dir": str(args.data_dir),
        "feature_count": len(feature_names),
        "features": feature_names,
        "labels": label_names,
        "train_size": len(train_features),
        "val_size": len(val_features),
        "test_size": len(test_features),
        "parameters": {
            "k_values": parse_int_list(args.k_values),
            "tree_depths": parse_int_list(args.tree_depths),
            "rf_depths": parse_int_list(args.rf_depths),
            "rf_trees": args.rf_trees,
            "bp_epochs": args.bp_epochs,
            "bp_hidden_size": args.bp_hidden_size,
            "bp_learning_rate": args.bp_learning_rate,
            "bp_l2": args.bp_l2,
            "bp_dropout_rate": args.bp_dropout_rate,
            "random_state": args.random_state,
        },
        "analysis": best_rows(rows),
        "outputs": {
            "summary_csv": str(summary_path),
            "accuracy_curves": str(accuracy_plot_path),
            "gap_curves": str(gap_plot_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Overfitting analysis finished")
    print(f"Summary: {summary_path}")
    print(f"Report:  {report_path}")
    print(f"Plots:   {accuracy_plot_path}, {gap_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
