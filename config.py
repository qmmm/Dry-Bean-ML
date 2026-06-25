"""全局路径、文件名和默认参数配置。"""

from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "DryBeanDataset"

SELECTED_STANDARDIZED_DIR = DATASET_DIR / "selected_features" / "standardized"
RESULTS_DIR = DATASET_DIR / "results"

SPLIT_FILES = {
    "train": "Dry_Bean_Dataset_Selected_Standardized_train.csv",
    "val": "Dry_Bean_Dataset_Selected_Standardized_val.csv",
    "test": "Dry_Bean_Dataset_Selected_Standardized_test.csv",
}

TARGET_COLUMN = "Class"

DEFAULT_RANDOM_STATE = 42
DEFAULT_KNN_K = 11

DEFAULT_ID3_PARAMS = {
    "max_depth": 12,
    "min_samples_leaf": 4,
    "min_samples_split": 8,
}

DEFAULT_RF_PARAMS = {
    "n_estimators": 41,
    "max_depth": 14,
    "min_samples_leaf": 3,
    "min_samples_split": 6,
    "bootstrap_ratio": 0.8,
}

DEFAULT_BP_PARAMS = {
    "hidden_size": 32,
    "epochs": 80,
    "learning_rate": 0.015,
    "l2": 5e-4,
    "dropout_rate": 0.15,
    "patience": 15,
}
