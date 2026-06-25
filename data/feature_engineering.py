#!/usr/bin/env python3
"""Create engineered Dry Bean features for classification models.

Input files are expected to come from data.clean. The script
writes raw engineered features and train-fitted standardized features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
from typing import Callable


DEFAULT_INPUT_DIR = Path("DryBeanDataset") / "cleaned"
DEFAULT_OUTPUT_DIR = Path("DryBeanDataset") / "features"

SPLIT_FILES = {
    "train": "Dry_Bean_Dataset_Clean_train.csv",
    "val": "Dry_Bean_Dataset_Clean_val.csv",
    "test": "Dry_Bean_Dataset_Clean_test.csv",
}

CLASS_COLUMN = "Class"
BASE_NUMERIC_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

LOG_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "ConvexArea",
    "EquivDiameter",
]


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def read_split(path: Path, split: str) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            converted: dict[str, float | str] = {"_split": split}
            for column in BASE_NUMERIC_COLUMNS:
                converted[column] = float(row[column])
            converted[CLASS_COLUMN] = row[CLASS_COLUMN]
            rows.append(converted)
        return rows


def add_engineered_features(row: dict[str, float | str]) -> dict[str, float | str]:
    features = dict(row)

    area = float(features["Area"])
    perimeter = float(features["Perimeter"])
    major_axis = float(features["MajorAxisLength"])
    minor_axis = float(features["MinorAxisLength"])
    aspect_ratio = float(features["AspectRation"])
    eccentricity = float(features["Eccentricity"])
    convex_area = float(features["ConvexArea"])
    equiv_diameter = float(features["EquivDiameter"])
    extent = float(features["Extent"])
    solidity = float(features["Solidity"])
    roundness = float(features["roundness"])
    compactness = float(features["Compactness"])

    axis_sum = major_axis + minor_axis
    axis_diff = major_axis - minor_axis
    convex_gap = convex_area - area
    ellipse_area = math.pi * major_axis * minor_axis / 4.0

    features.update(
        {
            "AspectRatio": aspect_ratio,
            "AxisLengthSum": axis_sum,
            "AxisLengthDiff": axis_diff,
            "AxisLengthMean": axis_sum / 2.0,
            "AxisDiffRatio": safe_divide(axis_diff, axis_sum),
            "MinorMajorRatio": safe_divide(minor_axis, major_axis),
            "AreaPerPerimeter": safe_divide(area, perimeter),
            "AreaPerPerimeterSquared": safe_divide(area, perimeter**2),
            "PerimeterPerArea": safe_divide(perimeter, area),
            "AreaConvexRatio": safe_divide(area, convex_area),
            "ConvexAreaGap": convex_gap,
            "ConvexAreaGapRatio": safe_divide(convex_gap, convex_area),
            "AreaEllipseRatio": safe_divide(area, ellipse_area),
            "EquivDiameterMajorRatio": safe_divide(equiv_diameter, major_axis),
            "EquivDiameterMinorRatio": safe_divide(equiv_diameter, minor_axis),
            "RoundnessSolidity": roundness * solidity,
            "CompactnessSolidity": compactness * solidity,
            "ExtentSolidity": extent * solidity,
            "EccentricityAspectRatio": eccentricity * aspect_ratio,
        }
    )

    for column in LOG_COLUMNS:
        features[f"log_{column}"] = math.log1p(float(features[column]))

    return features


def numeric_columns(rows: list[dict[str, float | str]]) -> list[str]:
    first_row = rows[0]
    return [
        column
        for column in first_row.keys()
        if column not in {"_split", CLASS_COLUMN}
    ]


def train_standardization_stats(
    rows: list[dict[str, float | str]],
    columns: list[str],
) -> dict[str, dict[str, float]]:
    train_rows = [row for row in rows if row["_split"] == "train"]
    stats: dict[str, dict[str, float]] = {}

    for column in columns:
        values = [float(row[column]) for row in train_rows]
        mean = fmean(values)
        std = pstdev(values)
        if std == 0:
            std = 1.0
        stats[column] = {"mean": mean, "std": std}

    return stats


def standardize_rows(
    rows: list[dict[str, float | str]],
    columns: list[str],
    stats: dict[str, dict[str, float]],
) -> list[dict[str, float | str]]:
    standardized_rows: list[dict[str, float | str]] = []

    for row in rows:
        standardized = {"_split": row["_split"]}
        for column in columns:
            standardized[column] = (
                float(row[column]) - stats[column]["mean"]
            ) / stats[column]["std"]
        standardized[CLASS_COLUMN] = row[CLASS_COLUMN]
        standardized_rows.append(standardized)

    return standardized_rows


def write_split(
    rows: list[dict[str, float | str]],
    split: str,
    columns: list[str],
    output_path: Path,
    transform_value: Callable[[float | str], float | str] | None = None,
) -> None:
    split_rows = [row for row in rows if row["_split"] == split]
    fieldnames = columns + [CLASS_COLUMN]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in split_rows:
            output_row = {}
            for column in columns:
                value = row[column]
                output_row[column] = transform_value(value) if transform_value else value
            output_row[CLASS_COLUMN] = row[CLASS_COLUMN]
            writer.writerow(output_row)


def class_distribution(rows: list[dict[str, float | str]], split: str) -> dict[str, int]:
    return dict(
        Counter(
            str(row[CLASS_COLUMN])
            for row in rows
            if row["_split"] == split
        ).most_common()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows: list[dict[str, float | str]] = []
    for split, file_name in SPLIT_FILES.items():
        input_path = args.input_dir / file_name
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        rows.extend(read_split(input_path, split))

    engineered_rows = [add_engineered_features(row) for row in rows]
    feature_columns = numeric_columns(engineered_rows)
    stats = train_standardization_stats(engineered_rows, feature_columns)
    standardized_rows = standardize_rows(engineered_rows, feature_columns, stats)

    raw_dir = args.output_dir / "raw"
    standardized_dir = args.output_dir / "standardized"
    raw_dir.mkdir(parents=True, exist_ok=True)
    standardized_dir.mkdir(parents=True, exist_ok=True)

    raw_files: dict[str, str] = {}
    standardized_files: dict[str, str] = {}
    for split in SPLIT_FILES:
        raw_path = raw_dir / f"Dry_Bean_Dataset_Features_{split}.csv"
        standardized_path = standardized_dir / f"Dry_Bean_Dataset_Features_Standardized_{split}.csv"
        write_split(engineered_rows, split, feature_columns, raw_path)
        write_split(standardized_rows, split, feature_columns, standardized_path)
        raw_files[split] = str(raw_path)
        standardized_files[split] = str(standardized_path)

    report = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "input_rows": dict(Counter(row["_split"] for row in rows)),
        "feature_count": len(feature_columns),
        "base_feature_count": len(BASE_NUMERIC_COLUMNS),
        "engineered_feature_count": len(feature_columns) - len(BASE_NUMERIC_COLUMNS),
        "base_features": BASE_NUMERIC_COLUMNS,
        "engineered_features": [
            column for column in feature_columns if column not in BASE_NUMERIC_COLUMNS
        ],
        "target_column": CLASS_COLUMN,
        "raw_feature_files": raw_files,
        "standardized_feature_files": standardized_files,
        "standardization": {
            "method": "z_score",
            "fit_on": "train split only",
            "stats": stats,
        },
        "class_distribution": {
            split: class_distribution(engineered_rows, split) for split in SPLIT_FILES
        },
    }

    report_path = args.output_dir / "feature_engineering_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Engineered Dry Bean features")
    print(f"Input rows:             {report['input_rows']}")
    print(f"Feature count:          {report['feature_count']}")
    print(f"Engineered new count:   {report['engineered_feature_count']}")
    print(f"Report:                 {report_path}")
    for split in SPLIT_FILES:
        print(f"{split:5s} raw:            {raw_files[split]}")
        print(f"{split:5s} standardized:   {standardized_files[split]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
