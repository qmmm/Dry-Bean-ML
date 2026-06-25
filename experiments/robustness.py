#!/usr/bin/env python3
"""比较不同算法在训练集噪声下的鲁棒性。"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from models.bp_nn import (
    BPNeuralNetwork,
    accuracy_score as bp_accuracy_score,
    encode_labels as encode_labels_for_bp,
    train_with_early_stopping,
)
from models.knn import KNNClassifier
from models.tree import (
    ID3DecisionTree,
    RandomForestClassifierFromScratch,
    accuracy_score,
    encode_labels,
    read_dataset,
)


DEFAULT_DATA_DIR = Path("DryBeanDataset") / "selected_features" / "standardized"
DEFAULT_OUTPUT_DIR = Path("DryBeanDataset") / "robustness_results"

SPLIT_FILES = {
    "train": "Dry_Bean_Dataset_Selected_Standardized_train.csv",
    "val": "Dry_Bean_Dataset_Selected_Standardized_val.csv",
    "test": "Dry_Bean_Dataset_Selected_Standardized_test.csv",
}

MODEL_ORDER = ["KNN", "ID3", "Random Forest", "BP Neural Network"]


def parse_float_list(raw_value: str) -> list[float]:
    values = []
    for part in raw_value.split(","):
        stripped = part.strip()
        if stripped:
            value = float(stripped)
            if value < 0.0:
                raise ValueError("noise intensities must be non-negative")
            values.append(value)
    return values


def stratified_limit(
    features: list[tuple[float, ...]],
    labels: list[str],
    limit: int,
    random_state: int,
) -> tuple[list[tuple[float, ...]], list[str]]:
    if limit <= 0 or limit >= len(features):
        return features, labels

    label_to_indices: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(index)

    rng = random.Random(random_state)
    selected: list[int] = []
    remaining = limit
    labels_left = len(label_to_indices)
    for label in sorted(label_to_indices):
        indices = label_to_indices[label][:]
        rng.shuffle(indices)
        take = min(len(indices), max(1, round(limit * len(indices) / len(labels))))
        if labels_left == 1:
            take = min(len(indices), remaining)
        selected.extend(indices[:take])
        remaining -= take
        labels_left -= 1

    if len(selected) < limit:
        selected_set = set(selected)
        extras = [index for index in range(len(features)) if index not in selected_set]
        rng.shuffle(extras)
        selected.extend(extras[: limit - len(selected)])

    selected = selected[:limit]
    selected.sort()
    return [features[index] for index in selected], [labels[index] for index in selected]


def add_gaussian_noise(
    features: list[tuple[float, ...]],
    intensity: float,
    rng: random.Random,
) -> list[tuple[float, ...]]:
    return [
        tuple(value + rng.gauss(0.0, intensity) for value in sample)
        for sample in features
    ]


def add_uniform_noise(
    features: list[tuple[float, ...]],
    intensity: float,
    rng: random.Random,
) -> list[tuple[float, ...]]:
    return [
        tuple(value + rng.uniform(-intensity, intensity) for value in sample)
        for sample in features
    ]


def add_feature_dropout_noise(
    features: list[tuple[float, ...]],
    intensity: float,
    rng: random.Random,
) -> list[tuple[float, ...]]:
    return [
        tuple(0.0 if rng.random() < intensity else value for value in sample)
        for sample in features
    ]


def add_label_flip_noise(
    labels: list[str],
    intensity: float,
    rng: random.Random,
) -> list[str]:
    label_names = sorted(set(labels))
    noisy_labels = labels[:]
    for index, label in enumerate(noisy_labels):
        if rng.random() < intensity:
            candidates = [candidate for candidate in label_names if candidate != label]
            noisy_labels[index] = rng.choice(candidates)
    return noisy_labels


def make_noisy_training_set(
    features: list[tuple[float, ...]],
    labels: list[str],
    noise_type: str,
    intensity: float,
    random_state: int,
) -> tuple[list[tuple[float, ...]], list[str]]:
    rng = random.Random(random_state)
    noisy_features = features
    noisy_labels = labels[:]

    if noise_type == "clean":
        return features, labels[:]
    if noise_type == "gaussian":
        noisy_features = add_gaussian_noise(features, intensity, rng)
    elif noise_type == "uniform":
        noisy_features = add_uniform_noise(features, intensity, rng)
    elif noise_type == "feature_dropout":
        noisy_features = add_feature_dropout_noise(features, intensity, rng)
    elif noise_type == "label_flip":
        noisy_labels = add_label_flip_noise(labels, intensity, rng)
    else:
        raise ValueError(f"unsupported noise type: {noise_type}")

    return noisy_features, noisy_labels


def load_best_knn_k(default: int = 11) -> int:
    report_path = Path("DryBeanDataset") / "knn_results" / "knn_report.json"
    if not report_path.exists():
        return default
    with report_path.open("r", encoding="utf-8") as handle:
        return int(json.load(handle).get("best_k", default))


def evaluate_knn(
    train_features: list[tuple[float, ...]],
    train_labels: list[str],
    test_features: list[tuple[float, ...]],
    test_labels: list[str],
    k: int,
) -> float:
    classifier = KNNClassifier(train_features, train_labels)
    predictions = classifier.predict_many(test_features, k)
    correct = sum(
        1 for expected, predicted in zip(test_labels, predictions) if expected == predicted
    )
    return correct / len(test_labels)


def evaluate_tree_models(
    train_features: list[tuple[float, ...]],
    train_labels_raw: list[str],
    val_labels_raw: list[str],
    test_features: list[tuple[float, ...]],
    test_labels_raw: list[str],
    random_state: int,
    rf_trees: int,
) -> tuple[float, float]:
    train_labels, _, test_labels, _ = encode_labels(
        train_labels_raw,
        val_labels_raw,
        test_labels_raw,
    )

    id3 = ID3DecisionTree(
        max_depth=12,
        min_samples_leaf=4,
        min_samples_split=8,
        random_state=random_state,
    )
    id3.fit(train_features, train_labels)
    id3_accuracy = accuracy_score(test_labels, id3.predict_many(test_features))

    random_forest = RandomForestClassifierFromScratch(
        n_estimators=rf_trees,
        max_depth=14,
        min_samples_leaf=3,
        min_samples_split=6,
        bootstrap_ratio=0.8,
        random_state=random_state,
    )
    random_forest.fit(train_features, train_labels)
    rf_accuracy = accuracy_score(test_labels, random_forest.predict_many(test_features))
    return id3_accuracy, rf_accuracy


def evaluate_bp_network(
    train_features: list[tuple[float, ...]],
    train_labels_raw: list[str],
    val_features: list[tuple[float, ...]],
    val_labels_raw: list[str],
    test_features: list[tuple[float, ...]],
    test_labels_raw: list[str],
    random_state: int,
    epochs: int,
    patience: int,
    show_epochs: bool,
) -> float:
    train_labels, val_labels, test_labels, label_names = encode_labels_for_bp(
        train_labels_raw,
        val_labels_raw,
        test_labels_raw,
    )
    network = BPNeuralNetwork(
        input_size=len(train_features[0]),
        hidden_size=32,
        output_size=len(label_names),
        learning_rate=0.015,
        l2=5e-4,
        dropout_rate=0.15,
        random_state=random_state,
    )
    if show_epochs:
        train_with_early_stopping(
            network,
            train_features,
            train_labels,
            val_features,
            val_labels,
            epochs=epochs,
            patience=patience,
            random_state=random_state,
        )
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            train_with_early_stopping(
                network,
                train_features,
                train_labels,
                val_features,
                val_labels,
                epochs=epochs,
                patience=patience,
                random_state=random_state,
            )
    return bp_accuracy_score(test_labels, network.predict_many(test_features))


def scenario_rows(
    noise_type: str,
    intensity: float,
    accuracies: dict[str, float],
    baseline: dict[str, float],
) -> list[dict]:
    rows = []
    for model in MODEL_ORDER:
        accuracy_value = accuracies[model]
        baseline_value = baseline[model]
        rows.append(
            {
                "noise_type": noise_type,
                "noise_intensity": intensity,
                "model": model,
                "test_accuracy": accuracy_value,
                "baseline_accuracy": baseline_value,
                "accuracy_drop": baseline_value - accuracy_value,
                "relative_drop_percent": (
                    (baseline_value - accuracy_value) / baseline_value * 100.0
                    if baseline_value
                    else 0.0
                ),
            }
        )
    return rows


def save_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "noise_type",
        "noise_intensity",
        "model",
        "test_accuracy",
        "baseline_accuracy",
        "accuracy_drop",
        "relative_drop_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_accuracy_drop_plot(rows: list[dict], output_path: Path) -> None:
    scenario_labels = []
    scenario_to_rows: dict[str, list[dict]] = {}
    for row in rows:
        if row["noise_type"] == "clean":
            continue
        label = f"{row['noise_type']}\n{row['noise_intensity']:.2f}"
        if label not in scenario_to_rows:
            scenario_labels.append(label)
            scenario_to_rows[label] = []
        scenario_to_rows[label].append(row)

    x_positions = list(range(len(scenario_labels)))
    width = 0.18
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    plt.figure(figsize=(13, 7))
    for model_index, model in enumerate(MODEL_ORDER):
        offsets = [
            position + (model_index - (len(MODEL_ORDER) - 1) / 2) * width
            for position in x_positions
        ]
        values = []
        for label in scenario_labels:
            row = next(item for item in scenario_to_rows[label] if item["model"] == model)
            values.append(row["accuracy_drop"])
        plt.bar(offsets, values, width=width, label=model, color=colors[model_index])

    plt.axhline(0.0, color="#444444", linewidth=0.8)
    plt.xticks(x_positions, scenario_labels, rotation=0)
    plt.ylabel("Accuracy drop versus clean training")
    plt.title("Robustness Comparison Under Training-Set Noise")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_accuracy_plot(rows: list[dict], output_path: Path) -> None:
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for row in rows:
        grouped.setdefault(row["noise_type"], {}).setdefault(row["model"], []).append(
            (row["noise_intensity"], row["test_accuracy"])
        )

    noise_types = [noise_type for noise_type in grouped if noise_type != "clean"]
    cols = 2
    rows_count = math.ceil(len(noise_types) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(13, 5 * rows_count), squeeze=False)
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    for axis, noise_type in zip(axes.flatten(), noise_types):
        for model, color in zip(MODEL_ORDER, colors):
            points = sorted(grouped[noise_type][model])
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                linewidth=1.8,
                label=model,
                color=color,
            )
        axis.set_title(noise_type)
        axis.set_xlabel("Noise intensity")
        axis.set_ylabel("Test accuracy")
        axis.grid(True, alpha=0.25)
        axis.set_ylim(0.0, 1.0)

    for axis in axes.flatten()[len(noise_types) :]:
        axis.axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(MODEL_ORDER))
    fig.suptitle("Clean-Test Accuracy After Noisy Training", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_scenario(
    train_features: list[tuple[float, ...]],
    train_labels: list[str],
    val_features: list[tuple[float, ...]],
    val_labels: list[str],
    test_features: list[tuple[float, ...]],
    test_labels: list[str],
    noise_type: str,
    intensity: float,
    random_state: int,
    knn_k: int,
    rf_trees: int,
    bp_epochs: int,
    bp_patience: int,
    show_bp_epochs: bool,
) -> dict[str, float]:
    noisy_features, noisy_labels = make_noisy_training_set(
        train_features,
        train_labels,
        noise_type,
        intensity,
        random_state,
    )

    print(f"Running {noise_type} intensity={intensity:.3f}")
    knn_accuracy = evaluate_knn(
        noisy_features,
        noisy_labels,
        test_features,
        test_labels,
        knn_k,
    )
    id3_accuracy, rf_accuracy = evaluate_tree_models(
        noisy_features,
        noisy_labels,
        val_labels,
        test_features,
        test_labels,
        random_state,
        rf_trees,
    )
    bp_accuracy = evaluate_bp_network(
        noisy_features,
        noisy_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        random_state,
        bp_epochs,
        bp_patience,
        show_bp_epochs,
    )

    return {
        "KNN": knn_accuracy,
        "ID3": id3_accuracy,
        "Random Forest": rf_accuracy,
        "BP Neural Network": bp_accuracy,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--noise-types",
        default="gaussian,uniform,feature_dropout,label_flip",
        help="Comma-separated noise types: gaussian, uniform, feature_dropout, label_flip.",
    )
    parser.add_argument(
        "--noise-intensities",
        default="0.30,0.50",
        help="Comma-separated noise strengths.",
    )
    parser.add_argument("--test-limit", type=int, default=900)
    parser.add_argument("--rf-trees", type=int, default=21)
    parser.add_argument("--bp-epochs", type=int, default=35)
    parser.add_argument("--bp-patience", type=int, default=8)
    parser.add_argument("--show-bp-epochs", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    noise_types = [part.strip() for part in args.noise_types.split(",") if part.strip()]
    noise_intensities = parse_float_list(args.noise_intensities)
    if not noise_types:
        raise ValueError("--noise-types cannot be empty")
    if not noise_intensities:
        raise ValueError("--noise-intensities cannot be empty")

    train_path = args.data_dir / SPLIT_FILES["train"]
    val_path = args.data_dir / SPLIT_FILES["val"]
    test_path = args.data_dir / SPLIT_FILES["test"]

    feature_names, train_features, train_labels = read_dataset(train_path)
    val_feature_names, val_features, val_labels = read_dataset(val_path)
    test_feature_names, test_features, test_labels = read_dataset(test_path)
    if feature_names != val_feature_names or feature_names != test_feature_names:
        raise ValueError("Feature columns differ between splits")

    test_features, test_labels = stratified_limit(
        test_features,
        test_labels,
        args.test_limit,
        args.random_state,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    knn_k = load_best_knn_k()
    baseline = run_scenario(
        train_features,
        train_labels,
        val_features,
        val_labels,
        test_features,
        test_labels,
        "clean",
        0.0,
        args.random_state,
        knn_k,
        args.rf_trees,
        args.bp_epochs,
        args.bp_patience,
        args.show_bp_epochs,
    )

    rows = scenario_rows("clean", 0.0, baseline, baseline)
    scenario_reports = {
        "clean_0.00": baseline,
    }

    for noise_type in noise_types:
        for intensity in noise_intensities:
            scenario_seed = args.random_state + int(intensity * 1000) + len(noise_type) * 97
            accuracies = run_scenario(
                train_features,
                train_labels,
                val_features,
                val_labels,
                test_features,
                test_labels,
                noise_type,
                intensity,
                scenario_seed,
                knn_k,
                args.rf_trees,
                args.bp_epochs,
                args.bp_patience,
                args.show_bp_epochs,
            )
            rows.extend(scenario_rows(noise_type, intensity, accuracies, baseline))
            scenario_key = f"{noise_type}_{intensity:.2f}"
            scenario_reports[scenario_key] = accuracies

    summary_path = args.output_dir / "robustness_summary.csv"
    report_path = args.output_dir / "robustness_report.json"
    drop_plot_path = args.output_dir / "accuracy_drop_by_noise.png"
    accuracy_plot_path = args.output_dir / "accuracy_by_noise.png"

    save_summary_csv(summary_path, rows)
    save_accuracy_drop_plot(rows, drop_plot_path)
    save_accuracy_plot(rows, accuracy_plot_path)

    report = {
        "data_dir": str(args.data_dir),
        "feature_count": len(feature_names),
        "features": feature_names,
        "train_size": len(train_features),
        "val_size": len(val_features),
        "test_size_used": len(test_features),
        "test_limit": args.test_limit,
        "noise_types": noise_types,
        "noise_intensities": noise_intensities,
        "model_order": MODEL_ORDER,
        "parameters": {
            "knn_k": knn_k,
            "rf_trees": args.rf_trees,
            "bp_epochs": args.bp_epochs,
            "bp_patience": args.bp_patience,
            "show_bp_epochs": args.show_bp_epochs,
            "random_state": args.random_state,
        },
        "baseline": baseline,
        "scenarios": scenario_reports,
        "outputs": {
            "summary_csv": str(summary_path),
            "accuracy_drop_plot": str(drop_plot_path),
            "accuracy_plot": str(accuracy_plot_path),
        },
    }
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Robustness comparison finished")
    print(f"Summary: {summary_path}")
    print(f"Report:  {report_path}")
    print(f"Plots:   {drop_plot_path}, {accuracy_plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
