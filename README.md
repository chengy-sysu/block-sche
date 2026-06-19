# BlockSche

BlockSche is a small framework for converting ordinary CUDA kernels into
persistent kernels whose logical thread blocks are represented as runtime tasks
(`rTask`). It is designed to sit after a CUDA code generator such as NNFusion:
the generated CUDA still looks like a normal `__global__` kernel, then
`block_sche_compiler.py` rewrites it into a persistent rTask kernel.

The runtime API follows the same general shape as CuSync's schedule objects:
the host builds a schedule buffer, uploads it once, and device code consumes
that schedule to recover the logical `blockIdx` that the original kernel used.

## Components

- `include/block_sche/block_sche.cuh`: header-only CUDA runtime.
- `block_sche_compiler.py`: source-to-source compiler CLI.
- `block_sche/compiler/transform.py`: Python compiler implementation.
- `examples/vector_add.cu`: ordinary CUDA input kernel.
- `examples/vector_add_launch.cu`: host-side schedule construction and launch.
- `tests/test_transform.py`: compiler contract tests.

## Source-to-Source Flow

Input:

```cuda
__global__ void vector_add(float* out, const float* a, const float* b, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}
```

Compile:

```bash
python3 block_sche_compiler.py examples/vector_add.cu \
  --kernel vector_add \
  --output examples/vector_add_persistent.cu
```

Generated shape:

```cuda
__device__ __forceinline__ void vector_add_rtask(
    block_sche::RTask __bs_task, block_sche::DeviceSchedule __bs_schedule,
    float* out, const float* a, const float* b, int n) {
  const uint3 __bs_blockIdx = {
      __bs_task.block.x, __bs_task.block.y, __bs_task.block.z};
  const dim3 __bs_gridDim(__bs_schedule.logical_grid.x,
      __bs_schedule.logical_grid.y, __bs_schedule.logical_grid.z);
  #define blockIdx __bs_blockIdx
  #define gridDim __bs_gridDim
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  ...
  #undef gridDim
  #undef blockIdx
}

__global__ void vector_add_persistent(
    block_sche::DeviceSchedule __bs_schedule, float* out,
    const float* a, const float* b, int n) {
  const uint32_t __bs_cta = block_sche::persistent_cta_id();
  const uint32_t __bs_stride = __bs_schedule.resident_ctas == 0
      ? gridDim.x * gridDim.y * gridDim.z
      : __bs_schedule.resident_ctas;
  for (uint32_t __bs_task_id = __bs_cta;
       __bs_task_id < __bs_schedule.task_count;
       __bs_task_id += __bs_stride) {
    block_sche::RTask __bs_task = __bs_schedule.tasks[__bs_task_id];
    vector_add_rtask(__bs_task, __bs_schedule, out, a, b, n);
  }
}
```

The compiler preserves `threadIdx`, `blockDim`, shared memory, and normal kernel
parameters. In the generated rTask body it creates local logical aliases:

- `blockIdx` resolves to the current `RTask`'s logical block coordinates.
- `gridDim` resolves to the original logical grid dimensions.

Add `--host-launcher` to emit a host wrapper around the generated persistent
kernel:

```bash
python3 block_sche_compiler.py examples/vector_add.cu \
  --kernel vector_add \
  --host-launcher \
  --output examples/vector_add_persistent.cu
```

The generated wrapper has this shape:

```cuda
cudaError_t vector_add_launch(block_sche::PreparedSchedule& prepared,
                              dim3 block_dim,
                              cudaStream_t stream,
                              float* out, const float* a,
                              const float* b, int n);
```

With `--trace`, the wrapper also takes `block_sche::PreparedTrace&` and resets
the trace buffer before launch.

For NNFusion-style generated files containing more than one CUDA kernel, use
`--all`:

```bash
python3 block_sche_compiler.py generated.cu \
  --all \
  --host-launcher \
  --output generated_persistent.cu
```

`--all` emits one `_rtask`, one `_persistent` kernel, and optionally one
`_launch` wrapper for each `__global__` kernel in the input file. It cannot be
combined with `--kernel` or `--output-kernel`.

## Mapping API

Build a host schedule with a logical grid, SM count, CTAs per SM, a block order,
and an SM mapper:

```cuda
auto schedule = block_sche::ScheduleBuilder(
                    block_sche::Dim3u(grid_x, grid_y, grid_z))
                    .sms(sm_count)
                    .ctas_per_sm(1)
                    .round_robin();
```

Equivalent lower-level API:

```cuda
auto schedule = block_sche::make_schedule(
    block_sche::Dim3u(grid_x, grid_y, grid_z),
    sm_count,
    1,
    block_sche::IdentityBlockOrder(),
    block_sche::RoundRobinSM());
```

`RTask` stores:

- `block`: logical CUDA block coordinates.
- `linear_block`: row-major logical block id.
- `sm`: requested software SM assignment.

`HostSchedule` also materializes per-SM task queues. `DeviceScheduleBuffer`
uploads:

- the flat `RTask` array
- `sm_offsets[sm]..sm_offsets[sm+1]` queue ranges
- `sm_task_ids` entries pointing back into the flat task array
- `sm_cursors` counters used by resident CTAs on the same hardware SM

For launches, `PreparedSchedule` owns the host schedule plus uploaded device
schedule:

```cuda
block_sche::PreparedSchedule prepared;
prepared.upload(schedule);
auto launch = prepared.launch_config(dim3(256, 1, 1));
kernel_persistent<<<launch.persistent_grid, launch.block_dim>>>(
    prepared.device_view(), ...);
```

The built-in mappers are:

- `RoundRobinSM`: `linear_block % sm_count`
- `BlockedSM`: contiguous runs of logical blocks per SM

The built-in block orders are:

- `IdentityBlockOrder`: row-major CUDA block order.
- `ColumnMajorBlockOrder`: y-major ordering within each z slice.
- `ExplicitBlockOrder`: caller-provided logical block order array.

Direct explicit block-to-SM placement:

```cuda
std::vector<uint32_t> block_to_sm = {0, 0, 1, 1, 2, 2};
auto schedule = block_sche::ScheduleBuilder(block_sche::Dim3u(6, 1, 1))
                    .sms(3)
                    .explicit_sm(block_to_sm);
```

Custom mappers implement:

```cuda
struct MyMapper {
  __host__ uint32_t operator()(uint32_t linear_block,
                               block_sche::Dim3u logical_grid,
                               uint32_t sm_count) const;
};
```

For strict SM-affine execution, compile with `--sm-affine`. The generated
kernel reads the hardware `%smid`, atomically claims entries from that SM's
queue, and only runs tasks assigned to that SM. If a schedule buffer is reused
for multiple launches, call `schedule_buffer.reset_cursors(stream)` before each
SM-affine relaunch.

For validation, compile with `--trace` as well. The persistent kernel receives a
`block_sche::DeviceTrace` parameter and records the hardware SM that executed
each logical block:

```cuda
block_sche::TraceBuffer trace;
trace.allocate(schedule.task_count());
kernel_persistent<<<...>>>(schedule_buffer.device_view(), trace.device_view(), ...);
std::vector<uint32_t> block_to_observed_sm;
trace.download(&block_to_observed_sm);
```

`PreparedTrace` is the equivalent RAII wrapper used by the tests.

## Current Limitations

The compiler can transform one named kernel or every kernel in a file with
`--all`. It expects concrete `__global__` kernels, including templated kernels,
dependent type parameters such as `typename Config::T*`, and common
`__launch_bounds__` placements used by modern Ampere examples. It virtualizes
`blockIdx` and `gridDim` in the generated rTask body, but it does not yet handle
kernels with complex macro-generated signatures, cooperative grid
synchronization, or device-side CUDA launches.

This keeps the first implementation suitable for NNFusion-style generated CUDA
while leaving room for a Clang LibTooling frontend later.

## Roadmap

- Add an optional fused persistent-launch mode for multiple kernels. The current
  compiler maps one original `__global__` kernel to one persistent kernel launch.
  A future mode can follow NNFusion BlockFusion's model: launch one fused kernel
  with the maximum block size needed by the participating kernels, pass a
  logical `thread_id` into each per-kernel device body, guard off threads beyond
  that kernel's original block size, and reconstruct that kernel's original
  `threadIdx`, `blockDim`, `blockIdx`, and `gridDim` inside the device body.
  This would allow kernels with different block sizes to share one persistent
  launch while preserving each kernel's original indexing semantics.
- Harden the compiler frontend. The current source-to-source compiler is meant
  for concrete NNFusion-style generated CUDA and is not a full CUDA parser. A
  Clang LibTooling frontend would make macro-generated signatures, attributes,
  namespaces, more complex templates, and unusual qualifiers much safer to
  transform.
- Add a thin NNFusion integration path after code generation. The core compiler
  should remain usable as a standalone src2src tool, but an integration wrapper
  can discover generated CUDA kernels, run the transform, and update host launch
  sites without requiring manual commands.
- Improve launch metadata handling. Today the caller is responsible for passing
  the original block size to the persistent launch. Generated launch wrappers can
  be extended to preserve per-kernel grid, block, stream, and shared-memory
  metadata more explicitly.
- Extend schedule validation and diagnostics. Host-side checks should report
  invalid SM ids, mismatched `block_to_sm` sizes, empty per-SM queues, and trace
  mismatches in a reusable way. SM-affine behavior should continue to be
  validated from observed `%smid` traces instead of inferred from CUDA launch
  placement.
- Handle or explicitly reject more CUDA features. Cooperative grid
  synchronization, dynamic parallelism, device-side launches, and kernels with
  complex synchronization assumptions need either first-class support or clearer
  compile-time diagnostics.
- Add broader examples and tests, including multi-kernel generated files, 2D/3D
  grids and blocks, dynamic shared memory, reductions, tiled matrix operations,
  and SM-affine trace validation under less uniform schedules.

## Verification

Run the compiler tests:

```bash
python3 -m pytest -q
```

Build the CUDA tests:

```bash
cmake -S . -B /tmp/block_sche_build
cmake --build /tmp/block_sche_build -j
ctest --test-dir /tmp/block_sche_build --output-on-failure
```

`vector_add_e2e` generates a persistent kernel from `examples/vector_add.cu`,
builds it with the runtime API, and launches it when a CUDA device is available.
`vector_add_sm_affine_e2e` additionally generates with `--sm-affine --trace`
and verifies that every logical block was executed on its requested SM.
When a `LeetCUDA/` checkout exists at the repository root,
`leetcuda_elementwise_e2e` and `leetcuda_sgemm_e2e` extract real kernels from
`LeetCUDA/kernels/elementwise/elementwise.cu` and
`LeetCUDA/kernels/sgemm/sgemm.cu`, run them through the src2src compiler, and
validate the generated persistent kernels on GPU. On machines without a visible
CUDA device, CTest marks those runtime tests as skipped.
