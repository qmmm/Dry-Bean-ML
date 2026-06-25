#!/usr/bin/env python3
"""Dry Bean 项目的统一命令入口。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CommandSpec:
    name: str
    module: str
    description: str
    fixed_args: tuple[str, ...] = ()


COMMANDS: dict[str, CommandSpec] = {
    "clean": CommandSpec("clean", "data.clean", "清洗 Dirty 数据"),
    "features": CommandSpec("features", "data.feature_engineering", "生成特征工程数据"),
    "select": CommandSpec("select", "data.select_features", "筛选特征"),
    "knn": CommandSpec("knn", "experiments.train_eval", "训练并评估 KNN", ("knn",)),
    "tree": CommandSpec("tree", "experiments.train_eval", "训练并评估 ID3 和随机森林", ("tree",)),
    "bp": CommandSpec("bp", "experiments.train_eval", "训练并评估 BP 神经网络", ("bp",)),
    "train": CommandSpec("train", "experiments.train_eval", "训练全部主模型", ("all",)),
    "compare": CommandSpec("compare", "experiments.compare_models", "多模型指标和图表对比"),
    "speed": CommandSpec("speed", "experiments.inference_speed", "推理速度测试"),
    "robustness": CommandSpec("robustness", "experiments.robustness", "鲁棒性实验"),
    "overfitting": CommandSpec("overfitting", "experiments.overfitting", "过拟合分析"),
    "sample-size": CommandSpec("sample-size", "experiments.sample_size", "样本规模适应性实验"),
}

GROUPS: dict[str, list[str]] = {
    "pipeline": ["clean", "features", "select"],
    "train": ["train"],
    "evaluate": ["compare", "speed"],
    "experiments": ["robustness", "overfitting"],
    "all": ["clean", "features", "select", "train", "compare", "speed"],
}


def run_command(spec: CommandSpec, extra_args: list[str], dry_run: bool) -> None:
    command = [
        sys.executable,
        "-m",
        spec.module,
        *spec.fixed_args,
        *extra_args,
    ]
    print(f"\n[{spec.name}] {spec.description}")
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def print_commands() -> None:
    print("Available commands:")
    for name in sorted(COMMANDS):
        spec = COMMANDS[name]
        fixed = " ".join(spec.fixed_args)
        target = f"{spec.module} {fixed}".strip()
        print(f"  {name:12s} {target:36s} {spec.description}")
    print("\nAvailable groups:")
    for name, commands in GROUPS.items():
        print(f"  {name:12s} {' -> '.join(commands)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry Bean 项目统一入口：数据处理、训练、评估和实验。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不真正运行")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    subparsers.add_parser("list", help="列出命令")

    run_parser = subparsers.add_parser("run", help="运行单个任务")
    run_parser.add_argument("name", choices=sorted(COMMANDS))
    run_parser.add_argument("task_args", nargs=argparse.REMAINDER)

    group_parser = subparsers.add_parser("group", help="运行一组任务")
    group_parser.add_argument("name", choices=sorted(GROUPS))

    return parser


def normalize_remainder(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "list":
        print_commands()
        return 0

    if args.mode == "run":
        run_command(COMMANDS[args.name], normalize_remainder(args.task_args), args.dry_run)
        return 0

    if args.mode == "group":
        for command_name in GROUPS[args.name]:
            run_command(COMMANDS[command_name], [], args.dry_run)
        return 0

    parser.error(f"Unsupported mode: {args.mode}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
