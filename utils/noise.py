"""鲁棒性实验噪声工具。"""

from __future__ import annotations

import random


def add_gaussian_noise(features: list[tuple[float, ...]], intensity: float, rng: random.Random) -> list[tuple[float, ...]]:
    return [tuple(value + rng.gauss(0.0, intensity) for value in sample) for sample in features]


def add_uniform_noise(features: list[tuple[float, ...]], intensity: float, rng: random.Random) -> list[tuple[float, ...]]:
    return [tuple(value + rng.uniform(-intensity, intensity) for value in sample) for sample in features]


def add_feature_dropout_noise(features: list[tuple[float, ...]], intensity: float, rng: random.Random) -> list[tuple[float, ...]]:
    return [tuple(0.0 if rng.random() < intensity else value for value in sample) for sample in features]


def add_label_flip_noise(labels: list[str], intensity: float, rng: random.Random) -> list[str]:
    label_names = sorted(set(labels))
    noisy_labels = labels[:]
    for index, label in enumerate(noisy_labels):
        if rng.random() < intensity:
            candidates = [candidate for candidate in label_names if candidate != label]
            noisy_labels[index] = rng.choice(candidates)
    return noisy_labels
