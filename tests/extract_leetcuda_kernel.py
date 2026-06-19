#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from block_sche.compiler.transform import TransformError, _find_kernel


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a self-contained LeetCUDA kernel slice for BlockSche e2e tests."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        source = args.input.read_text()
        kernel = _find_kernel(source, args.kernel)
    except (OSError, TransformError) as exc:
        print(f"extract_leetcuda_kernel: {exc}", file=sys.stderr)
        return 1

    # Keep the real LeetCUDA prelude and selected kernel, but drop Torch/PyBind
    # headers so the slice can be compiled by plain nvcc in CTest.
    sliced = source[: kernel.body_end + 1]
    lines = [
        line
        for line in sliced.splitlines(keepends=True)
        if "<torch/" not in line and "torch/types.h" not in line
    ]
    output = "".join(lines)
    if not output.endswith("\n"):
        output += "\n"
    args.output.write_text(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
