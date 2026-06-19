# BlockSche

BlockSche 是一个 CUDA source-to-source 编译器和轻量级 runtime。它把普通
`__global__` kernel 改写成 persistent kernel，并把原来的逻辑 CUDA block
表示成 runtime task，也就是 `RTask`。这样 host 端可以显式构造 block 执行顺序、
block 到 SM 的软件映射，并在 GPU 上由 persistent CTA 消费这些任务。

一句话总结：BlockSche 将普通 CUDA kernel 转换为可调度的 persistent kernel，
让逻辑 thread block 可以按用户定义的顺序和 SM 映射执行。

## 设计目标

- 保持输入 kernel 尽量接近普通 CUDA/代码生成器输出。
- 提供独立的 src2src 编译器，而不是绑定到某个前端框架。
- 在 runtime 中显式表达逻辑 block 顺序和 block-to-SM 映射。
- 用 trace 验证实际执行 SM，而不是依赖对 CUDA block scheduler 行为的推测。
- 为 NNFusion 这类 CUDA 代码生成器后处理留出集成路径。

## 项目结构

- `block_sche_compiler.py`：src2src 编译器 CLI。
- `block_sche/compiler/transform.py`：Python 编译器实现。
- `include/block_sche/block_sche.cuh`：header-only CUDA runtime。
- `examples/vector_add.cu`：普通 CUDA 输入 kernel 示例。
- `examples/vector_add_launch.cu`：手写 host 端 schedule 和 launch 示例。
- `tests/test_transform.py`：编译器输出的 Python 回归测试。
- `tests/*_e2e_main.cu`：CUDA runtime/e2e 测试。
- `tests/extract_leetcuda_kernel.py`：从 LeetCUDA 源文件抽取单个 kernel slice
  用于 e2e。

`LeetCUDA/`、`cusync/`、`nnfusion/` 在当前工作区中作为参考代码树使用，不属于
BlockSche 核心源码。

## 快速开始

把普通 CUDA kernel 转成 persistent kernel：

```bash
python3 block_sche_compiler.py examples/vector_add.cu \
  --kernel vector_add \
  --output examples/vector_add_persistent.cu
```

如果输入文件里有多个 `__global__` kernel，可以一次转换全部：

```bash
python3 block_sche_compiler.py generated.cu \
  --all \
  --output generated_persistent.cu
```

如果希望同时生成 host launch wrapper：

```bash
python3 block_sche_compiler.py examples/vector_add.cu \
  --kernel vector_add \
  --host-launcher \
  --output examples/vector_add_persistent.cu
```

`--all` 会为每个原始 kernel 生成一组 `_rtask`、`_persistent`，以及可选的
`_launch` wrapper。它不能和 `--kernel` 或 `--output-kernel` 同时使用。

## Src2src 改写形态

输入 kernel：

```cuda
__global__ void vector_add(float* out, const float* a, const float* b, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}
```

生成代码的核心形态：

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

编译器只在 `_rtask` body 中虚拟化：

- `blockIdx`：来自当前 `RTask` 的逻辑 block 坐标。
- `gridDim`：来自 `DeviceSchedule.logical_grid` 的原始逻辑 grid。

编译器不会虚拟化：

- `threadIdx`
- `blockDim`
- warp/lane 相关内建变量
- shared memory 语义

因此，调用 persistent kernel 时必须传入原 kernel 期望的 block size。当前实现是
“一个原始 kernel 对应一个 persistent launch”，所以不同原始 kernel 有不同
block size 时，应分别生成和 launch。

## Runtime 和编译器接口

src2src 编译器和 runtime API 之间有一个很小的 ABI 约定：

- `block_sche::RTask`
- `block_sche::DeviceSchedule`
- `block_sche::DeviceTrace`
- `block_sche::current_sm_id()`
- `block_sche::record_trace()`

编译器生成的 persistent kernel 接收 `DeviceSchedule`。如果启用 `--trace`，还会
额外接收 `DeviceTrace`。host 端负责构造 schedule、上传到 GPU，并用原 kernel
对应的 block size 启动 persistent kernel。

## 构造调度

最常见的用法是通过 `ScheduleBuilder` 构造 host schedule：

```cuda
auto schedule = block_sche::ScheduleBuilder(
                    block_sche::Dim3u(grid_x, grid_y, grid_z))
                    .sms(sm_count)
                    .ctas_per_sm(1)
                    .round_robin();
```

等价的底层 API：

```cuda
auto schedule = block_sche::make_schedule(
    block_sche::Dim3u(grid_x, grid_y, grid_z),
    sm_count,
    1,
    block_sche::IdentityBlockOrder(),
    block_sche::RoundRobinSM());
```

`RTask` 中包含：

- `block`：逻辑 CUDA block 坐标。
- `linear_block`：row-major 线性 block id。
- `sm`：用户请求的软件 SM assignment。

`HostSchedule` 会同时生成扁平任务数组和 per-SM task queue。上传到 GPU 的
`DeviceSchedule` 包含：

- `tasks`：扁平 `RTask` 数组。
- `sm_offsets`：每个 SM 的 task queue 范围。
- `sm_task_ids`：per-SM queue 到 `tasks` 的索引。
- `sm_cursors`：SM-affine 模式下同一硬件 SM 上 resident CTA 的原子游标。

启动 persistent kernel：

```cuda
block_sche::PreparedSchedule prepared;
prepared.upload(schedule);

auto launch = prepared.launch_config(dim3(256, 1, 1));
vector_add_persistent<<<launch.persistent_grid, launch.block_dim>>>(
    prepared.device_view(), out, a, b, n);
```

## Block 到 SM 的映射

BlockSche 支持两层概念：

1. 逻辑 block 到软件 SM id 的映射，也就是 `RTask.sm`。
2. persistent CTA 实际运行在哪个硬件 SM 上，也就是 GPU thread block scheduler
   的分配结果。

内置 SM mapper：

- `RoundRobinSM`：`linear_block % sm_count`
- `BlockedSM`：连续若干个逻辑 block 映射到同一个 SM

内置 block order：

- `IdentityBlockOrder`：CUDA row-major block 顺序。
- `ColumnMajorBlockOrder`：每个 z slice 内按 y-major 顺序。
- `ExplicitBlockOrder`：调用者给出完整逻辑 block 顺序。

显式指定 block-to-SM：

```cuda
std::vector<uint32_t> block_to_sm = {0, 0, 1, 1, 2, 2};
auto schedule = block_sche::ScheduleBuilder(block_sche::Dim3u(6, 1, 1))
                    .sms(3)
                    .explicit_sm(block_to_sm);
```

自定义 mapper：

```cuda
struct MyMapper {
  __host__ uint32_t operator()(uint32_t linear_block,
                               block_sche::Dim3u logical_grid,
                               uint32_t sm_count) const;
};
```

非 `--sm-affine` 模式下，`RTask.sm` 只是调度元数据。persistent CTA 可以在任意
硬件 SM 上执行，具体由 GPU 上的 thread block scheduler 决定。

严格 SM-affine 执行需要编译时启用：

```bash
python3 block_sche_compiler.py input.cu \
  --kernel my_kernel \
  --sm-affine \
  --output output.cu
```

`--sm-affine` 生成的 persistent kernel 会读取硬件 `%smid`，然后只从该 SM 对应
的 software queue 中 claim task。也就是说，严格绑定不是靠“launch `sm_count`
个 CTA 就一定每个 SM 一个 CTA”实现的，而是靠实际落到某个 SM 的 persistent CTA
只消费该 SM 的队列实现的。

这里的 CTA/thread block 落点语义来自 GPU 的硬件 thread block scheduler。相关
依据可参考 Gilman et al. 的论文 *Demystifying the Placement Policies of the
NVIDIA GPU Thread Block Scheduler for Concurrent Kernels*。该工作通过实测说明
thread block scheduler 负责把 block 分配到 SM，并且分配会受每个 SM 的本地资源
可用性影响。BlockSche 的测试也只根据 `%smid` trace 判断实际执行位置。

如果复用同一个 schedule buffer 多次 launch，SM-affine 模式下需要在每次 relaunch
前重置游标：

```cuda
schedule_buffer.reset_cursors(stream);
```

## Trace 验证

启用 `--trace` 后，persistent kernel 会记录每个逻辑 block 实际运行的硬件 SM：

```bash
python3 block_sche_compiler.py input.cu \
  --kernel my_kernel \
  --sm-affine \
  --trace \
  --output output.cu
```

host 端使用：

```cuda
block_sche::TraceBuffer trace;
trace.allocate(schedule.task_count());

kernel_persistent<<<...>>>(
    schedule_buffer.device_view(), trace.device_view(), ...);

std::vector<uint32_t> block_to_observed_sm;
trace.download(&block_to_observed_sm);
```

测试中也提供了 RAII 版本 `PreparedTrace`。

## 当前支持范围

编译器可以转换一个具名 kernel，也可以用 `--all` 转换文件中的所有具体
`__global__` kernel。当前支持：

- 普通 C 风格 kernel 参数。
- templated kernel。
- `typename Config::T*` 这类 dependent type 参数。
- LeetCUDA/Ampere kernel 中常见的 `__launch_bounds__` 写法。
- 多行 kernel 声明。
- 注释或字符串中的伪 `__global__`、`<<< >>>` 会被忽略。

当前不支持或只做显式拒绝：

- 宏生成的复杂 kernel signature。
- cooperative grid synchronization。
- device-side CUDA launch 和 dynamic parallelism。
- 需要跨 CTA 同步假设的 kernel。
- 自动改写 host 侧所有 launch metadata。
- 把多个不同 block size 的 kernel 合并到一个 persistent launch。

## LeetCUDA 覆盖

如果仓库根目录存在 `LeetCUDA/` checkout，CMake 会自动启用两个真实 LeetCUDA
e2e：

- `leetcuda_elementwise_e2e`
- `leetcuda_sgemm_e2e`

这两个测试会从 LeetCUDA 源文件抽取真实 kernel，经过 `block_sche_compiler.py`
生成 persistent kernel，再在 GPU 上验证结果。

当前 src2src 前端也用 LeetCUDA 全仓库 kernel 做过 transform smoke：79 个文件、
293 个 kernel、0 个 transform 失败。这个 smoke 只证明前端能完成转换，不等价于
每个 kernel 都已经有数值正确性的 e2e。

## 构建和测试

运行 Python 编译器测试：

```bash
python3 -m pytest -q
```

构建 CUDA 测试：

```bash
cmake -S . -B /tmp/block_sche_build
cmake --build /tmp/block_sche_build -j
```

运行 CTest：

```bash
ctest --test-dir /tmp/block_sche_build --output-on-failure
```

当前测试包含：

- `schedule_layout_test`：host schedule layout 测试。
- `vector_add_e2e`：普通 vector add src2src 到 GPU e2e。
- `vector_add_sm_affine_e2e`：`--sm-affine --trace`，验证每个逻辑 block 的实际
  `%smid`。
- `leetcuda_elementwise_e2e`：LeetCUDA elementwise kernel e2e。
- `leetcuda_sgemm_e2e`：LeetCUDA SGEMM kernel e2e。

如果机器没有可见 CUDA device，CUDA runtime 测试会返回 CTest skip code `77`。
在受限沙箱中看不到 `/dev/nvidia*` 时，`no CUDA-capable device is detected`
通常是设备访问限制，不应直接判断为代码错误。

## Roadmap

- 多 kernel fused persistent launch：参考 NNFusion BlockFusion 的方向，用一个
  fused kernel 承载多个原始 kernel。需要为每个原始 kernel 保存自己的
  `threadIdx`、`blockDim`、`blockIdx`、`gridDim` 语义，并 guard 掉超过原 block
  size 的线程。
- 更强的 CUDA 前端：当前实现是工程化 src2src 扫描器，不是完整 CUDA parser。
  后续可以引入 Clang LibTooling 来处理宏、namespace、复杂 template、attribute
  和更多 qualifier。
- NNFusion 后处理集成：保持核心编译器独立，同时增加 wrapper 自动发现生成的
  CUDA kernel、运行转换、更新 host launch。
- launch metadata 管理：让生成代码更完整地保存原始 grid、block、stream、
  dynamic shared memory 等信息，减少调用者手工传参。
- schedule 诊断：增加 invalid SM id、`block_to_sm` 尺寸不匹配、空队列、trace
  mismatch 等 host 侧错误报告。
- 更广泛 e2e：覆盖 2D/3D grid 和 block、dynamic shared memory、reduction、
  tiled matrix kernel、更多 LeetCUDA Ampere kernel、非均匀 SM-affine schedule。
- 更清晰的 unsupported diagnostics：对 cooperative groups、dynamic parallelism、
  device-side launch、特殊同步假设给出更具体的编译期错误。
