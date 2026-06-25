"""绘图公共设置。"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def save_line_plot(path, x_values, series, title: str, xlabel: str, ylabel: str) -> None:
    plt.figure(figsize=(10, 6))
    for label, values in series.items():
        plt.plot(x_values, values, marker="o", linewidth=1.8, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
