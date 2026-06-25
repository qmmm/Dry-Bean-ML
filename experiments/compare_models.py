#!/usr/bin/env python3
"""Visualize model results for the Dry Bean classification experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DATASET_DIR = Path("DryBeanDataset")
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "visualizations"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_model_reports(dataset_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """读取前面各个算法脚本输出的结果报告。"""
    knn = load_json(dataset_dir / "knn_results" / "knn_report.json")
    tree = load_json(dataset_dir / "tree_model_results" / "tree_model_report.json")
    bp = load_json(dataset_dir / "bp_nn_results" / "bp_nn_report.json")

    reports = {
        "KNN": knn["test_report"],
        "ID3": tree["models"]["id3"]["test_report"],
        "Random Forest": tree["models"]["random_forest"]["test_report"],
        "BP Neural Network": bp["test_report"],
    }
    labels = knn["test_report"]["labels"]
    return reports, labels


def metrics_rows(reports: dict[str, dict]) -> list[dict]:
    rows = []
    for model_name, report in reports.items():
        rows.append(
            {
                "model": model_name,
                "accuracy": report["accuracy"],
                "macro_precision": report["macro_avg"]["precision"],
                "macro_recall": report["macro_avg"]["recall"],
                "macro_f1": report["macro_avg"]["f1"],
                "weighted_precision": report["weighted_avg"]["precision"],
                "weighted_recall": report["weighted_avg"]["recall"],
                "weighted_f1": report["weighted_avg"]["f1"],
            }
        )
    return rows


def per_class_rows(reports: dict[str, dict]) -> list[dict]:
    rows = []
    for model_name, report in reports.items():
        for label, values in report["per_class"].items():
            rows.append(
                {
                    "model": model_name,
                    "class": label,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1"],
                    "support": values["support"],
                }
            )
    return rows


def save_confusion_matrices(
    reports: dict[str, dict],
    labels: list[str],
    output_dir: Path,
) -> dict[str, str]:
    """绘制每个算法的混淆矩阵，并额外生成一个 2x2 总览图。"""
    paths: dict[str, str] = {}
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(17, 14))
    axes_flat = axes.flatten()

    for axis, (model_name, report) in zip(axes_flat, reports.items()):
        matrix = pd.DataFrame(report["confusion_matrix"], index=labels, columns=labels)
        normalized = matrix.div(matrix.sum(axis=1), axis=0)

        sns.heatmap(
            normalized,
            ax=axis,
            cmap="YlGnBu",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            square=True,
            cbar=False,
            linewidths=0.4,
        )
        axis.set_title(f"{model_name} confusion matrix")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)

        single_path = cm_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        plt.figure(figsize=(8.5, 7.2))
        sns.heatmap(
            matrix,
            cmap="YlGnBu",
            annot=True,
            fmt="d",
            square=True,
            linewidths=0.4,
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title(f"{model_name} confusion matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(single_path, dpi=220)
        plt.close()
        paths[f"{model_name}_confusion_matrix"] = str(single_path)

    fig.suptitle("Normalized Test Confusion Matrices", fontsize=16, y=0.995)
    fig.tight_layout()
    grid_path = cm_dir / "all_models_confusion_matrices.png"
    fig.savefig(grid_path, dpi=220)
    plt.close(fig)
    paths["all_models_confusion_matrices"] = str(grid_path)
    return paths


def save_overall_metric_bars(metrics_df: pd.DataFrame, output_dir: Path) -> str:
    """绘制 Accuracy、Precision、Recall、F1 的算法对比分组柱状图。"""
    plot_df = metrics_df.melt(
        id_vars="model",
        value_vars=["accuracy", "macro_precision", "macro_recall", "macro_f1"],
        var_name="metric",
        value_name="score",
    )
    metric_names = {
        "accuracy": "Accuracy",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
    }
    plot_df["metric"] = plot_df["metric"].map(metric_names)

    output_path = output_dir / "overall_precision_recall_f1_grouped_bar.png"
    plt.figure(figsize=(12, 7))
    axis = sns.barplot(data=plot_df, x="model", y="score", hue="metric", palette="Set2")
    axis.set_ylim(0.86, 1.01)
    axis.set_title("Overall Test Metrics by Model")
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.tick_params(axis="x", rotation=12)

    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return str(output_path)


def save_per_class_metric_bars(per_class_df: pd.DataFrame, output_dir: Path) -> str:
    """绘制每个类别的 F1 对比，帮助定位哪些豆类最难分。"""
    output_path = output_dir / "per_class_f1_grouped_bar.png"
    plt.figure(figsize=(14, 7))
    axis = sns.barplot(
        data=per_class_df,
        x="class",
        y="f1",
        hue="model",
        palette="tab10",
    )
    axis.set_ylim(0.82, 1.01)
    axis.set_title("Per-Class Test F1 by Model")
    axis.set_xlabel("Class")
    axis.set_ylabel("F1")
    axis.tick_params(axis="x", rotation=20)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return str(output_path)


def save_bp_loss_curve(dataset_dir: Path, output_dir: Path) -> str:
    """BP 是迭代优化模型，因此这里绘制训练/验证 loss 曲线。"""
    history_path = dataset_dir / "bp_nn_results" / "bp_nn_training_history.csv"
    history = pd.read_csv(history_path)

    output_path = output_dir / "bp_nn_loss_curve.png"
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_loss"], marker="o", linewidth=1.6, label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], marker="s", linewidth=1.6, label="Validation loss")
    best_epoch = int(history.loc[history["val_accuracy"].idxmax(), "epoch"])
    plt.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1.2, label=f"Best epoch {best_epoch}")
    plt.title("BP Neural Network Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return str(output_path)


def save_bp_accuracy_curve(dataset_dir: Path, output_dir: Path) -> str:
    history_path = dataset_dir / "bp_nn_results" / "bp_nn_training_history.csv"
    history = pd.read_csv(history_path)

    output_path = output_dir / "bp_nn_accuracy_curve.png"
    plt.figure(figsize=(10, 6))
    plt.plot(history["epoch"], history["train_accuracy"], marker="o", linewidth=1.6, label="Train accuracy")
    plt.plot(history["epoch"], history["val_accuracy"], marker="s", linewidth=1.6, label="Validation accuracy")
    best_epoch = int(history.loc[history["val_accuracy"].idxmax(), "epoch"])
    plt.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1.2, label=f"Best epoch {best_epoch}")
    plt.title("BP Neural Network Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    return str(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports, labels = collect_model_reports(args.dataset_dir)
    metrics_df = pd.DataFrame(metrics_rows(reports))
    per_class_df = pd.DataFrame(per_class_rows(reports))

    metrics_path = args.output_dir / "model_visualization_summary.csv"
    per_class_path = args.output_dir / "per_class_metric_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    per_class_df.to_csv(per_class_path, index=False)

    output_paths = {}
    output_paths.update(save_confusion_matrices(reports, labels, args.output_dir))
    output_paths["overall_metric_grouped_bar"] = save_overall_metric_bars(metrics_df, args.output_dir)
    output_paths["per_class_f1_grouped_bar"] = save_per_class_metric_bars(per_class_df, args.output_dir)
    output_paths["bp_loss_curve"] = save_bp_loss_curve(args.dataset_dir, args.output_dir)
    output_paths["bp_accuracy_curve"] = save_bp_accuracy_curve(args.dataset_dir, args.output_dir)

    summary = {
        "models": list(reports.keys()),
        "labels": labels,
        "note": (
            "KNN, ID3, and Random Forest are non-gradient/non-iterative in this project, "
            "so only the BP neural network has train/validation loss curves."
        ),
        "metric_summary_csv": str(metrics_path),
        "per_class_metric_summary_csv": str(per_class_path),
        "plots": output_paths,
        "best_test_accuracy_model": metrics_df.sort_values(
            "accuracy", ascending=False
        ).iloc[0].to_dict(),
    }
    summary_path = args.output_dir / "visualization_report.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Model visualizations finished")
    print(f"Summary CSV: {metrics_path}")
    print(f"Report:      {summary_path}")
    for name, path in output_paths.items():
        print(f"{name}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
