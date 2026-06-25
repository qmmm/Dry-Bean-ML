"""读取 CSV、划分数据和标签编码。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from config import SELECTED_STANDARDIZED_DIR, SPLIT_FILES, TARGET_COLUMN


@dataclass
class DatasetSplit:
    feature_names: list[str]
    features: list[tuple[float, ...]]
    labels: list[str]


@dataclass
class DatasetBundle:
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


def read_dataset(path: Path) -> DatasetSplit:
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

    return DatasetSplit(feature_names, features, labels)


def load_splits(data_dir: Path = SELECTED_STANDARDIZED_DIR) -> DatasetBundle:
    bundle = DatasetBundle(
        train=read_dataset(data_dir / SPLIT_FILES["train"]),
        val=read_dataset(data_dir / SPLIT_FILES["val"]),
        test=read_dataset(data_dir / SPLIT_FILES["test"]),
    )
    ensure_same_features(bundle)
    return bundle


def ensure_same_features(bundle: DatasetBundle) -> None:
    feature_names = bundle.train.feature_names
    if bundle.val.feature_names != feature_names or bundle.test.feature_names != feature_names:
        raise ValueError("Feature columns differ between splits")


def encode_labels(*label_sets: list[str]) -> tuple[list[list[int]], list[str]]:
    label_names = sorted(set().union(*[set(labels) for labels in label_sets]))
    label_to_index = {label: index for index, label in enumerate(label_names)}
    encoded = [[label_to_index[label] for label in labels] for labels in label_sets]
    return encoded, label_names
