# AGENTS.md

This file is for coding agents that continue work in this repository. Keep it
short, factual, and aligned with the current implementation.

## Project Scope

BlockSche is a CUDA source-to-source compiler plus a small header-only runtime
API. It converts ordinary `__global__` kernels into persistent kernels whose
logical CUDA blocks are represented as runtime tasks (`RTask`). The runtime can
build software schedules that map logical blocks to requested SM ids.

Keep the core project focused on:

- the src2src compiler
- the runtime mapping/schedule API
- examples and tests proving the generated kernels work

Do not reintroduce a CMake helper/package abstraction unless explicitly asked.
The current CMake file intentionally uses direct `add_custom_command` calls.

## Important Paths

- `include/block_sche/block_sche.cuh`: header-only runtime API.
- `block_sche_compiler.py`: compiler CLI entry point.
- `block_sche/compiler/transform.py`: CUDA transform implementation.
- `examples/vector_add.cu`: simple input kernel.
- `examples/vector_add_launch.cu`: manual host-side schedule/launch example.
- `tests/test_transform.py`: compiler contract tests.
- `tests/*_e2e_main.cu`: CUDA runtime/e2e tests.
- `CMakeLists.txt`: builds generated CUDA tests with direct commands.
- `README.md`: user-facing usage, limitations, roadmap, and verification.

`cusync/` and `nnfusion/` are reference trees in this workspace. Treat them as
read-only unless the user explicitly asks for changes there.

## Current Architecture

The compiler and runtime are intentionally coupled through a small ABI:

- `block_sche::RTask`
- `block_sche::DeviceSchedule`
- `block_sche::DeviceTrace` when `--trace` is used
- helper functions such as `current_sm_id()` and `record_trace()`

The compiler generates:

- one device rTask body per input kernel
- one persistent kernel per input kernel
- optional host launch wrapper with `--host-launcher`

`--all` transforms every concrete `__global__` kernel in a file, but each
original kernel still becomes its own persistent launch.

Only `blockIdx` and `gridDim` are virtualized in the generated rTask body.
`threadIdx` and `blockDim` are the real CUDA values from the persistent launch.
Therefore callers must launch the persistent kernel with the same block size the
original kernel expected.

`--sm-affine` is a software scheduling mode. The generated kernel reads hardware
`%smid`, claims tasks from that SM's queue, and records/uses only tasks assigned
to that SM. Do not claim that launching `sm_count` CTAs guarantees one CTA per
SM. Validate SM placement from trace data.

## Mapping API

Host code builds schedules with `ScheduleBuilder` or `make_schedule`.

Common options:

- `.round_robin()`
- `.blocked(block_run)`
- `.column_major_round_robin()`
- `.explicit_sm(block_to_sm)`
- `.custom(block_order, sm_mapper)`

Explicit mappings are block-to-SM metadata and become strict only with
`--sm-affine`. In non-SM-affine mode, `RTask.sm` is metadata; CUDA may execute
the persistent CTA on any hardware SM.

## Known Limitations

Keep README and tests honest about these limits:

- The compiler is not a full CUDA parser.
- Macro-generated signatures and unusual declarations may fail.
- Cooperative grid synchronization is not supported.
- Device-side CUDA launches and dynamic parallelism are not rewritten.
- Different original block sizes cannot currently be fused into one persistent
  launch.
- Launch metadata such as original grid/block/shared-memory settings is still
  mostly caller-managed.

The README roadmap tracks likely next work: a stronger frontend, NNFusion
post-codegen integration, fused persistent launches, better diagnostics, and
broader tests.

## Development Rules

- Preserve the current narrow scope unless the user explicitly expands it.
- Prefer simple src2src/compiler/runtime changes over new build abstractions.
- Keep generated CUDA readable and deterministic.
- Add focused tests for compiler output when changing transform logic.
- Add CUDA e2e coverage when changing runtime behavior.
- Do not modify unrelated untracked reference trees.
- Do not infer CUDA scheduler behavior without trace evidence.

## Verification

Run Python compiler tests:

```bash
python3 -m pytest -q
```

Build CUDA tests:

```bash
cmake -S . -B /tmp/block_sche_build
cmake --build /tmp/block_sche_build -j
```

Run CTest:

```bash
ctest --test-dir /tmp/block_sche_build --output-on-failure
```

In restricted sandboxes, CUDA tests may not see `/dev/nvidia*` and can report
`no CUDA-capable device is detected`. If GPU validation is required, request
escalation for:

```bash
ctest --test-dir /tmp/block_sche_build --output-on-failure
```

with a narrowly scoped prefix such as:

```json
["ctest", "--test-dir"]
```

Do not treat sandbox GPU invisibility as a code failure unless the same command
also fails on the host with GPU access.
