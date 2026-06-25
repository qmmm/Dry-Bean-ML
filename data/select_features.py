#!/usr/bin/env python3
"""Select a compact Dry Bean feature set.

Selection is fitted on the training split only:
1. Remove highly correlated redundant features.
2. Export the remaining features for train/val/test in raw and standardized forms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_FEATURE_DIR = Path("DryBeanDataset") / "features"
DEFAULT_OUTPUT_DIR = Path("DryBeanDataset") / "selected_features"

SPLITS = ["train", "val", "test"]
TARGET_COLUMN = "Class"


def load_feature_splits(feature_dir: Path, variant: str) -> dict[str, pd.DataFrame]:
    if variant == "raw":
        pattern = "Dry_Bean_Dataset_Features_{split}.csv"
    elif variant == "standardized":
        pattern = "Dry_Bean_Dataset_Features_Standardized_{split}.csv"
    else:
        raise ValueError(f"Unknown feature variant: {variant}")

    frames = {}
    for split in SPLITS:
        path = feature_dir / variant / pattern.format(split=split)
        if not path.exists():
            raise FileNotFoundError(path)
        frames[split] = pd.read_csv(path)
    return frames


def save_heatmap(correlation: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(18, 15))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.1,
        cbar_kws={"shrink": 0.75},
    )
    plt.title(title)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def remove_correlated_features(
    train_features: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], list[dict[str, float | str]], pd.DataFrame]:
    correlation = train_features.corr(method="pearson")
    abs_correlation = correlation.abs()
    kept_features: list[str] = []
    dropped_features: list[dict[str, float | str]] = []

    for feature in train_features.columns:
        redundant_with = None
        redundant_corr = 0.0
        for kept_feature in kept_features:
            feature_corr = float(abs_correlation.loc[feature, kept_feature])
            if feature_corr >= threshold:
                redundant_with = kept_feature
                redundant_corr = feature_corr
                break

        if redundant_with is None:
            kept_features.append(feature)
        else:
            dropped_features.append(
                {
                    "dropped": feature,
                    "kept": redundant_with,
                    "abs_correlation": redundant_corr,
                }
            )

    return kept_features, dropped_features, correlation


def export_selected_splits(
    frames: dict[str, pd.DataFrame],
    selected_features: list[str],
    output_dir: Path,
    variant: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for split, frame in frames.items():
        selected_frame = frame[selected_features + [TARGET_COLUMN]]
        if variant == "raw":
            file_name = f"Dry_Bean_Dataset_Selected_{split}.csv"
        else:
            file_name = f"Dry_Bean_Dataset_Selected_Standardized_{split}.csv"
        output_path = output_dir / file_name
        selected_frame.to_csv(output_path, index=False)
        output_files[split] = str(output_path)
    return output_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--corr-threshold", type=float, default=0.95)
    args = parser.parse_args()

    raw_frames = load_feature_splits(args.feature_dir, "raw")
    standardized_frames = load_feature_splits(args.feature_dir, "standardized")
    feature_columns = [
        column for column in raw_frames["train"].columns if column != TARGET_COLUMN
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    train_features = raw_frames["train"][feature_columns]
    kept_after_corr, dropped_by_corr, correlation = remove_correlated_features(
        train_features=train_features,
        threshold=args.corr_threshold,
    )

    heatmap_path = plots_dir / "train_feature_correlation_heatmap.png"
    save_heatmap(
        correlation,
        heatmap_path,
        f"Train Feature Correlation Heatmap ({len(feature_columns)} features)",
    )

    selected_features = kept_after_corr

    raw_output_files = export_selected_splits(
        raw_frames,
        selected_features,
        args.output_dir / "raw",
        "raw",
    )
    standardized_output_files = export_selected_splits(
        standardized_frames,
        selected_features,
        args.output_dir / "standardized",
        "standardized",
    )

    report = {
        "method": [
            "Pearson correlation redundancy removal on train split",
            "Correlation-selected feature export for train/val/test",
        ],
        "target_column": TARGET_COLUMN,
        "correlation_threshold": args.corr_threshold,
        "initial_feature_count": len(feature_columns),
        "kept_after_correlation_count": len(kept_after_corr),
        "dropped_by_correlation_count": len(dropped_by_corr),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "dropped_by_correlation": dropped_by_corr,
        "plots": {
            "correlation_heatmap": str(heatmap_path),
        },
        "raw_selected_files": raw_output_files,
        "standardized_selected_files": standardized_output_files,
    }

    report_path = args.output_dir / "feature_selection_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Selected Dry Bean features")
    print(f"Initial features:       {len(feature_columns)}")
    print(f"After correlation:      {len(kept_after_corr)}")
    print(f"Dropped by correlation: {len(dropped_by_corr)}")
    print(f"Selected features:      {len(selected_features)}")
    print(f"Report:                 {report_path}")
    print(f"Heatmap:                {heatmap_path}")
    print("Selected feature names:")
    for feature in selected_features:
        print(f"  - {feature}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
