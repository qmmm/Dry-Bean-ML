#!/usr/bin/env python3
"""Benchmark inference speed for the Dry Bean classifiers.

The timed part includes prediction only. Data loading and model fitting are done
before timing starts, so the result focuses on inference speed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from models.bp_nn import (
    BPNeuralNetwork,
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
DEFAULT_OUTPUT_DIR = Path("DryBeanDataset") / "inference_speed_results"

SPLIT_FILES = {
    "train": "Dry_Bean_Dataset_Selected_Standardized_train.csv",
    "val": "Dry_Bean_Dataset_Selected_Standardized_val.csv",
    "test": "Dry_Bean_Dataset_Selected_Standardized_test.csv",
}


def load_best_knn_k(default: int = 11) -> int:
    report_path = Path("DryBeanDataset") / "knn_results" / "knn_report.json"
    if not report_path.exists():
        return default
    with report_path.open("r", encoding="utf-8") as handle:
        return int(json.load(handle).get("best_k", default))


def benchmark_predictor(
    name: str,
    predict_fn: Callable[[], list],
    expected_count: int,
    repeats: int,
    warmups: int,
) -> dict:
    """重复计时 predict 函数；只记录推理耗时，不记录训练耗时。"""
    for _ in range(warmups):
        predictions = predict_fn()
        if len(predictions) != expected_count:
            raise ValueError(f"{name} returned {len(predictions)} predictions")

    timings = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            start = time.perf_counter()
            predictions = predict_fn()
            elapsed = time.perf_counter() - start
            if len(predictions) != expected_count:
                raise ValueError(f"{name} returned {len(predictions)} predictions")
            timings.append(elapsed)
    finally:
        if gc_was_enabled:
            gc.enable()

    mean_seconds = statistics.fmean(timings)
    median_seconds = statistics.median(timings)
    min_seconds = min(timings)
    max_seconds = max(timings)
    p95_seconds = sorted(timings)[max(0, int(len(timings) * 0.95) - 1)]
    ms_per_sample = mean_seconds / expected_count * 1000.0
    samples_per_second = expected_count / mean_seconds

    return {
        "model": name,
        "repeats": repeats,
        "test_samples": expected_count,
        "mean_total_seconds": mean_seconds,
        "median_total_seconds": median_seconds,
        "min_total_seconds": min_seconds,
        "max_total_seconds": max_seconds,
        "p95_total_seconds": p95_seconds,
        "mean_ms_per_sample": ms_per_sample,
        "samples_per_second": samples_per_second,
        "all_total_seconds": timings,
    }


def save_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "test_accuracy",
        "repeats",
        "test_samples",
        "mean_total_seconds",
        "median_total_seconds",
        "min_total_seconds",
        "max_total_seconds",
        "p95_total_seconds",
        "mean_ms_per_sample",
        "samples_per_second",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def save_total_time_bar(rows: list[dict], output_path: Path) -> None:
    models = [row["model"] for row in rows]
    values = [row["mean_total_seconds"] for row in rows]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=["#6db1bf", "#e08d79", "#8fa6c9", "#9fc27a"])
    plt.yscale("log")
    plt.ylabel("Mean total inference time on test set (seconds, log scale)")
    plt.title("Total Test-Set Inference Time by Model")
    plt.xticks(rotation=10)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}s",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_ms_per_sample_bar(rows: list[dict], output_path: Path) -> None:
    models = [row["model"] for row in rows]
    values = [row["mean_ms_per_sample"] for row in rows]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=["#6db1bf", "#e08d79", "#8fa6c9", "#9fc27a"])
    plt.yscale("log")
    plt.ylabel("Mean inference latency (ms/sample, log scale)")
    plt.title("Single-Sample Inference Latency by Model")
    plt.xticks(rotation=10)
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative")

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

    print("Fitting models before timing inference...")

    knn_k = load_best_knn_k()
    knn = KNNClassifier(train_features, train_label_names)

    id3 = ID3DecisionTree(
        max_depth=12,
        min_samples_leaf=4,
        min_samples_split=8,
        random_state=args.random_state,
    )
    id3.fit(train_features, train_labels)

    random_forest = RandomForestClassifierFromScratch(
        n_estimators=41,
        max_depth=14,
        min_samples_leaf=3,
        min_samples_split=6,
        bootstrap_ratio=0.8,
        random_state=args.random_state,
    )
    random_forest.fit(train_features, train_labels)

    bp_train_labels, bp_val_labels, bp_test_labels, _ = encode_labels_for_bp(
        train_label_names,
        val_label_names,
        test_label_names,
    )
    bp = BPNeuralNetwork(
        input_size=len(feature_names),
        hidden_size=32,
        output_size=len(label_names),
        learning_rate=0.015,
        l2=5e-4,
        dropout_rate=0.15,
        random_state=args.random_state,
    )
    train_with_early_stopping(
        bp,
        train_features,
        bp_train_labels,
        val_features,
        bp_val_labels,
        epochs=80,
        patience=15,
        random_state=args.random_state,
    )

    predictors = [
        ("KNN", lambda: knn.predict_many(test_features, knn_k), test_label_names),
        ("ID3", lambda: id3.predict_many(test_features), test_labels),
        ("Random Forest", lambda: random_forest.predict_many(test_features), test_labels),
        ("BP Neural Network", lambda: bp.predict_many(test_features), bp_test_labels),
    ]

    rows = []
    for model_name, predict_fn, expected_labels in predictors:
        print(f"Benchmarking {model_name}...")
        result = benchmark_predictor(
            model_name,
            predict_fn,
            expected_count=len(test_features),
            repeats=args.repeats,
            warmups=args.warmups,
        )
        predictions = predict_fn()
        if model_name == "KNN":
            test_accuracy = sum(
                1 for expected, predicted in zip(expected_labels, predictions)
                if expected == predicted
            ) / len(expected_labels)
        else:
            test_accuracy = accuracy_score(expected_labels, predictions)
        result["test_accuracy"] = test_accuracy
        rows.append(result)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "inference_speed_summary.csv"
    report_json = args.output_dir / "inference_speed_report.json"
    total_time_plot = args.output_dir / "total_inference_time_bar.png"
    ms_per_sample_plot = args.output_dir / "ms_per_sample_bar.png"

    save_summary_csv(summary_csv, rows)
    save_total_time_bar(rows, total_time_plot)
    save_ms_per_sample_bar(rows, ms_per_sample_plot)

    report = {
        "data_dir": str(args.data_dir),
        "feature_count": len(feature_names),
        "test_samples": len(test_features),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "timing_scope": "prediction only; data loading and model fitting are excluded",
        "knn_k": knn_k,
        "results": rows,
        "outputs": {
            "summary_csv": str(summary_csv),
            "total_inference_time_bar": str(total_time_plot),
            "ms_per_sample_bar": str(ms_per_sample_plot),
        },
    }
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Inference speed benchmark finished")
    for row in rows:
        print(
            f"{row['model']}: total={row['mean_total_seconds']:.6f}s, "
            f"ms/sample={row['mean_ms_per_sample']:.6f}, "
            f"samples/sec={row['samples_per_second']:.2f}, "
            f"accuracy={row['test_accuracy']:.6f}"
        )
    print(f"Summary: {summary_csv}")
    print(f"Plot 1:  {total_time_plot}")
    print(f"Plot 2:  {ms_per_sample_plot}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
