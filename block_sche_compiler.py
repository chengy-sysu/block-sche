#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from block_sche.compiler import TransformOptions, transform_all_cuda_kernels, transform_cuda_source
from block_sche.compiler.transform import TransformError


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite a normal CUDA __global__ kernel into a BlockSche persistent rTask kernel."
    )
    parser.add_argument("input", type=Path, help="input .cu file")
    parser.add_argument("-o", "--output", type=Path, help="output .cu file; defaults to stdout")
    parser.add_argument("--kernel", help="kernel name to transform when the file has multiple kernels")
    parser.add_argument(
        "--all",
        action="store_true",
        help="transform every __global__ kernel in the input file",
    )
    parser.add_argument("--output-kernel", help="generated persistent kernel name")
    parser.add_argument(
        "--sm-affine",
        action="store_true",
        help="emit a persistent kernel that only executes tasks mapped to the resident SM",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="add a DeviceTrace parameter and record the hardware SM that executes each rTask",
    )
    parser.add_argument(
        "--host-launcher",
        action="store_true",
        help="emit a cudaError_t host wrapper that launches the generated persistent kernel",
    )
    parser.add_argument(
        "--drop-original",
        action="store_true",
        help="omit the original dynamic-scheduler kernel from the generated output",
    )
    parser.add_argument(
        "--no-runtime-include",
        action="store_true",
        help="do not add #include <block_sche/block_sche.cuh>",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        options = TransformOptions(
            kernel_name=args.kernel,
            output_kernel_name=args.output_kernel,
            include_runtime=not args.no_runtime_include,
            sm_affine=args.sm_affine,
            trace=args.trace,
            host_launcher=args.host_launcher,
            keep_original=not args.drop_original,
        )
        source = args.input.read_text()
        generated = (
            transform_all_cuda_kernels(source, options)
            if args.all
            else transform_cuda_source(source, options)
        )
    except (OSError, TransformError) as exc:
        print(f"block_sche_compiler: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.write_text(generated)
    else:
        sys.stdout.write(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
