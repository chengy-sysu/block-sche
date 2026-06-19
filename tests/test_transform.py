from pathlib import Path
import subprocess
import sys

import pytest

from block_sche.compiler import TransformOptions, transform_all_cuda_kernels, transform_cuda_source
from block_sche.compiler.transform import TransformError


ROOT = Path(__file__).resolve().parents[1]


def test_transform_rewrites_block_and_grid_indices():
    source = """
__global__ void saxpy(float* y, const float* x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  if (i < n) y[i] = a * x[i] + y[i];
}
"""

    generated = transform_cuda_source(source, TransformOptions(kernel_name="saxpy"))

    assert "#include <block_sche/block_sche.cuh>" in generated
    assert "__device__ __forceinline__ void saxpy_rtask" in generated
    assert "__global__ void saxpy_persistent(block_sche::DeviceSchedule __bs_schedule" in generated
    assert "for (uint32_t __bs_task_id = __bs_cta;" in generated
    assert "const uint3 __bs_blockIdx = {__bs_task.block.x, __bs_task.block.y, __bs_task.block.z};" in generated
    assert "const dim3 __bs_gridDim(__bs_schedule.logical_grid.x" in generated
    assert "#define blockIdx __bs_blockIdx" in generated
    assert "#define gridDim __bs_gridDim" in generated
    assert "int i = blockIdx.x * blockDim.x + threadIdx.x;" in generated
    assert "int stride = gridDim.x * blockDim.x;" in generated


def test_transform_can_emit_sm_affine_runner():
    source = """
__global__ void fill(float* y) {
  y[blockIdx.x] = gridDim.x;
}
"""

    generated = transform_cuda_source(
        source, TransformOptions(kernel_name="fill", sm_affine=True)
    )

    assert "const uint32_t __bs_sm = block_sche::current_sm_id();" in generated
    assert "__bs_schedule.sm_offsets[__bs_sm]" in generated
    assert "__shared__ uint32_t __bs_task_block_x;" in generated
    assert "const block_sche::RTask __bs_task = {block_sche::Dim3u" in generated
    assert "if (block_sche::is_block_leader())" in generated
    assert "atomicAdd(&__bs_schedule.sm_cursors[__bs_sm], 1)" in generated
    assert "__bs_schedule.sm_task_ids[__bs_claimed_pos]" in generated
    assert "y[blockIdx.x]" in generated
    assert "= gridDim.x;" in generated


def test_transform_can_emit_trace_parameter_and_record():
    source = """
__global__ void fill(float* y) {
  y[blockIdx.x] = gridDim.x;
}
"""

    generated = transform_cuda_source(
        source, TransformOptions(kernel_name="fill", sm_affine=True, trace=True)
    )

    assert "__global__ void fill_persistent(block_sche::DeviceSchedule __bs_schedule, block_sche::DeviceTrace __bs_trace" in generated
    assert "block_sche::record_trace(__bs_trace, __bs_task, __bs_sm);" in generated


def test_transform_can_emit_host_launcher():
    source = """
__global__ void fill(float* y, int n) {
  if (blockIdx.x < n) y[blockIdx.x] = gridDim.x;
}
"""

    generated = transform_cuda_source(
        source, TransformOptions(kernel_name="fill", host_launcher=True)
    )

    assert "cudaError_t fill_launch(block_sche::PreparedSchedule& __bs_prepared" in generated
    assert "cudaStream_t __bs_stream, float* y, int n)" in generated
    assert "__bs_prepared.reset_cursors(__bs_stream)" in generated
    assert "fill_persistent<<<__bs_launch.persistent_grid" in generated
    assert "(__bs_prepared.device_view(), y, n);" in generated


def test_transform_can_emit_traced_host_launcher():
    source = """
__global__ void fill(float* y) {
  y[blockIdx.x] = gridDim.x;
}
"""

    generated = transform_cuda_source(
        source,
        TransformOptions(
            kernel_name="fill", sm_affine=True, trace=True, host_launcher=True
        ),
    )

    assert "block_sche::PreparedTrace& __bs_trace" in generated
    assert "__bs_trace.reset(__bs_stream)" in generated
    assert "(__bs_prepared.device_view(), __bs_trace.device_view(), y);" in generated


def test_transform_supports_whole_blockidx_alias_use():
    source = """
__global__ void copy_block(uint3* out) {
  uint3 b = blockIdx;
  out[threadIdx.x] = b;
}
"""

    generated = transform_cuda_source(source, TransformOptions(kernel_name="copy_block"))

    assert "uint3 b = blockIdx;" in generated
    assert "#define blockIdx __bs_blockIdx" in generated


def test_transform_preserves_template_kernel_prefix():
    source = """
template <typename T>
__global__ void scale(T* y, T a) {
  y[blockIdx.x] *= a;
}
"""

    generated = transform_cuda_source(source, TransformOptions(kernel_name="scale"))

    assert "template <typename T>\n__device__ __forceinline__ void scale_rtask" in generated
    assert "template <typename T>\n__global__ void scale_persistent" in generated


def test_transform_rejects_unsupported_dynamic_parallelism():
    source = """
__global__ void parent(float* y) {
  child<<<1, 1>>>(y);
}
"""

    with pytest.raises(TransformError, match="device-side CUDA launches"):
        transform_cuda_source(source, TransformOptions(kernel_name="parent"))


def test_transform_all_cuda_kernels_emits_each_kernel():
    source = """
__global__ void add1(int* y) { y[blockIdx.x] += 1; }
__global__ void add2(int* y) { y[blockIdx.x] += 2; }
"""

    generated = transform_all_cuda_kernels(
        source, TransformOptions(host_launcher=True, keep_original=False)
    )

    assert "__global__ void add1(" not in generated
    assert "__global__ void add2(" not in generated
    assert "__global__ void add1_persistent" in generated
    assert "__global__ void add2_persistent" in generated
    assert "cudaError_t add1_launch" in generated
    assert "cudaError_t add2_launch" in generated


def test_transform_all_rejects_specific_kernel_options():
    with pytest.raises(TransformError, match="--all cannot be combined"):
        transform_all_cuda_kernels(
            "__global__ void k() {}",
            TransformOptions(kernel_name="k"),
        )


def test_transform_requires_kernel_name_for_multiple_kernels():
    source = """
__global__ void a() {}
__global__ void b() {}
"""
    with pytest.raises(TransformError, match="multiple __global__ kernels"):
        transform_cuda_source(source)


def test_cli_writes_transformed_file(tmp_path):
    src = tmp_path / "input.cu"
    out = tmp_path / "output.cu"
    src.write_text(
        "__global__ void k(int* y) { y[blockIdx.x + gridDim.x] = threadIdx.x; }\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "block_sche_compiler.py"),
            str(src),
            "--kernel",
            "k",
            "-o",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    generated = out.read_text()
    assert "__global__ void k_persistent" in generated
    assert "y[blockIdx.x + gridDim.x] = threadIdx.x;" in generated
    assert "const uint3 __bs_blockIdx" in generated
    assert "const dim3 __bs_gridDim" in generated


def test_cli_can_transform_all_kernels(tmp_path):
    src = tmp_path / "input.cu"
    out = tmp_path / "output.cu"
    src.write_text(
        "__global__ void a(int* y) { y[blockIdx.x] += 1; }\n"
        "__global__ void b(int* y) { y[blockIdx.x] += 2; }\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "block_sche_compiler.py"),
            str(src),
            "--all",
            "--drop-original",
            "--host-launcher",
            "-o",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    generated = out.read_text()
    assert "__global__ void a_persistent" in generated
    assert "__global__ void b_persistent" in generated
    assert "cudaError_t a_launch" in generated
    assert "cudaError_t b_launch" in generated
