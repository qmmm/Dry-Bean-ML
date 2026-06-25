#!/usr/bin/env python3
"""Clean the dirty Dry Bean dataset for downstream classification.

The script keeps the original files untouched and writes cleaned CSV files plus
an audit report under DryBeanDataset/cleaned by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


DEFAULT_INPUT_DIR = Path("DryBeanDataset")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "cleaned"

SPLIT_FILES = {
    "train": "Dry_Bean_Dataset_Dirty_train.csv",
    "val": "Dry_Bean_Dataset_Dirty_val.csv",
    "test": "Dry_Bean_Dataset_Dirty_test.csv",
}

NUMERIC_COLUMNS = [
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

CLASS_COLUMN = "Class"
OUTPUT_COLUMNS = NUMERIC_COLUMNS + [CLASS_COLUMN]

VALID_CLASSES = {
    "BARBUNYA",
    "BOMBAY",
    "CALI",
    "DERMASON",
    "HOROZ",
    "SEKER",
    "SIRA",
}

MISSING_TOKENS = {"", "?", "NA", "N/A", "NULL", "NONE", "NAN"}
UNIT_VALUE_PATTERN = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z]+)\s*$"
)


def read_split(path: Path, split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row_number, row in enumerate(reader, start=2):
            row["_split"] = split
            row["_source_row"] = str(row_number)
            rows.append(row)
        return rows


def normalize_class(raw_value: str | None) -> str:
    value = (raw_value or "").strip().upper()
    value = value.replace("3", "E").replace("0", "O")
    return value


def parse_numeric(raw_value: str | None) -> tuple[float | None, str | None]:
    value = (raw_value or "").strip()
    if value.upper() in MISSING_TOKENS:
        return None, "missing_token"

    try:
        parsed = float(value)
    except ValueError:
        unit_match = UNIT_VALUE_PATTERN.match(value)
        if unit_match:
            return float(unit_match.group(1)), f"removed_unit_{unit_match.group(2).lower()}"
        return None, "not_numeric"

    if math.isnan(parsed) or math.isinf(parsed):
        return None, "not_finite"
    return parsed, None


def clean_row(raw_row: dict[str, str], issue_counts: dict[str, Counter]) -> dict[str, Any]:
    split = raw_row["_split"]
    cleaned: dict[str, Any] = {
        "_split": split,
        "_source_row": raw_row["_source_row"],
    }

    normalized_class = normalize_class(raw_row.get(CLASS_COLUMN))
    if normalized_class not in VALID_CLASSES:
        issue_counts[split]["invalid_class"] += 1
        cleaned[CLASS_COLUMN] = ""
    else:
        if normalized_class != (raw_row.get(CLASS_COLUMN) or ""):
            issue_counts[split]["class_normalized"] += 1
        cleaned[CLASS_COLUMN] = normalized_class

    for column in NUMERIC_COLUMNS:
        parsed, parse_issue = parse_numeric(raw_row.get(column))
        if parse_issue:
            issue_counts[split][f"{column}:{parse_issue}"] += 1
        cleaned[column] = parsed

    # Physically impossible signs are treated as data-entry noise where safe.
    area = cleaned["Area"]
    if area is not None and area < 0:
        cleaned["Area"] = abs(area)
        issue_counts[split]["Area:negative_to_abs"] += 1

    # The dataset is image-derived; these measures should be positive.
    for column in NUMERIC_COLUMNS:
        value = cleaned[column]
        if value is not None and value <= 0:
            cleaned[column] = None
            issue_counts[split][f"{column}:non_positive_to_missing"] += 1

    for column in ["Eccentricity", "Extent", "Solidity", "roundness", "Compactness", "ShapeFactor3", "ShapeFactor4"]:
        value = cleaned[column]
        if value is not None and not 0 < value <= 1:
            cleaned[column] = None
            issue_counts[split][f"{column}:outside_0_1_to_missing"] += 1

    # Very large major axes are clear data-entry outliers in this dataset.
    major_axis = cleaned["MajorAxisLength"]
    if major_axis is not None and major_axis > 1000:
        cleaned["MajorAxisLength"] = None
        issue_counts[split]["MajorAxisLength:gt_1000_to_missing"] += 1

    area = cleaned["Area"]
    convex_area = cleaned["ConvexArea"]
    if area is not None and convex_area is not None and area > convex_area:
        cleaned["Area"] = None
        issue_counts[split]["Area:greater_than_convex_to_missing"] += 1

    return cleaned


def train_medians(rows: list[dict[str, Any]]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for column in NUMERIC_COLUMNS:
        values = [
            row[column]
            for row in rows
            if row["_split"] == "train" and isinstance(row[column], float)
        ]
        if not values:
            raise ValueError(f"Cannot compute a training median for {column}")
        medians[column] = float(median(values))
    return medians


def impute_missing(
    rows: list[dict[str, Any]],
    medians: dict[str, float],
    issue_counts: dict[str, Counter],
) -> None:
    for row in rows:
        split = row["_split"]
        for column, fill_value in medians.items():
            if row[column] is None:
                row[column] = fill_value
                issue_counts[split][f"{column}:imputed_train_median"] += 1


def row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[column] for column in OUTPUT_COLUMNS)


def deduplicate_rows(
    rows: list[dict[str, Any]],
    issue_counts: dict[str, Counter],
) -> list[dict[str, Any]]:
    deduped_by_split: list[dict[str, Any]] = []
    seen_per_split: dict[str, set[tuple[Any, ...]]] = defaultdict(set)

    for row in rows:
        split = row["_split"]
        signature = row_signature(row)
        if signature in seen_per_split[split]:
            issue_counts[split]["duplicate_within_split_dropped"] += 1
            continue
        seen_per_split[split].add(signature)
        deduped_by_split.append(row)

    eval_signatures = {
        row_signature(row)
        for row in deduped_by_split
        if row["_split"] in {"val", "test"}
    }
    final_rows = []
    for row in deduped_by_split:
        if row["_split"] == "train" and row_signature(row) in eval_signatures:
            issue_counts["train"]["duplicate_in_eval_dropped_from_train"] += 1
            continue
        final_rows.append(row)

    return final_rows


def write_split(rows: list[dict[str, Any]], split: str, output_dir: Path) -> Path:
    output_path = output_dir / f"Dry_Bean_Dataset_Clean_{split}.csv"
    split_rows = [row for row in rows if row["_split"] == split]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in split_rows:
            writer.writerow({column: row[column] for column in OUTPUT_COLUMNS})

    return output_path


def class_distribution(rows: list[dict[str, Any]], split: str | None = None) -> dict[str, int]:
    selected = rows if split is None else [row for row in rows if row["_split"] == split]
    return dict(Counter(row[CLASS_COLUMN] for row in selected).most_common())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    raw_rows: list[dict[str, str]] = []
    for split, file_name in SPLIT_FILES.items():
        path = args.input_dir / file_name
        if not path.exists():
            raise FileNotFoundError(path)
        raw_rows.extend(read_split(path, split))

    issue_counts: dict[str, Counter] = defaultdict(Counter)
    cleaned_rows = [clean_row(row, issue_counts) for row in raw_rows]
    medians = train_medians(cleaned_rows)
    impute_missing(cleaned_rows, medians, issue_counts)
    final_rows = deduplicate_rows(cleaned_rows, issue_counts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        split: str(write_split(final_rows, split, args.output_dir))
        for split in SPLIT_FILES
    }

    report = {
        "input_rows": dict(Counter(row["_split"] for row in raw_rows)),
        "output_rows": dict(Counter(row["_split"] for row in final_rows)),
        "output_files": output_files,
        "numeric_columns": NUMERIC_COLUMNS,
        "target_column": CLASS_COLUMN,
        "training_medians_used_for_imputation": medians,
        "class_distribution_after_cleaning": {
            split: class_distribution(final_rows, split) for split in SPLIT_FILES
        },
        "issue_counts": {
            split: dict(counter.most_common())
            for split, counter in sorted(issue_counts.items())
        },
    }

    report_path = args.output_dir / "cleaning_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Cleaned Dry Bean dataset")
    print(f"Input rows:  {report['input_rows']}")
    print(f"Output rows: {report['output_rows']}")
    print(f"Report:      {report_path}")
    for split, path in output_files.items():
        print(f"{split:5s}:       {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
