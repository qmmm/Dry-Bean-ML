"""单模型训练和测试入口。"""

from __future__ import annotations

import argparse
import subprocess
import sys


TRAIN_MODULES = {
    "knn": "models.knn",
    "tree": "models.tree",
    "bp": "models.bp_nn",
}


def run_module(module: str, args: list[str]) -> None:
    subprocess.run([sys.executable, "-m", module, *args], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=["knn", "tree", "bp", "all"])
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    extra_args = args.script_args[1:] if args.script_args[:1] == ["--"] else args.script_args
    algorithms = ["knn", "tree", "bp"] if args.algorithm == "all" else [args.algorithm]
    for algorithm in algorithms:
        run_module(TRAIN_MODULES[algorithm], extra_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
