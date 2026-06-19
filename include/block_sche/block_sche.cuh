#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <stdint.h>
#include <vector>

namespace block_sche {

constexpr uint32_t kInvalidSM = 0xffffffffu;

struct Dim3u {
  uint32_t x;
  uint32_t y;
  uint32_t z;

  __host__ __device__ Dim3u() : x(1), y(1), z(1) {}
  __host__ __device__ Dim3u(uint32_t x_, uint32_t y_, uint32_t z_ = 1)
      : x(x_), y(y_), z(z_) {}
  __host__ __device__ explicit Dim3u(dim3 d) : x(d.x), y(d.y), z(d.z) {}

  __host__ __device__ dim3 as_dim3() const { return dim3(x, y, z); }
};

struct RTask {
  Dim3u block;
  uint32_t linear_block;
  uint32_t sm;
};

struct DeviceSchedule {
  const RTask* tasks;
  const uint32_t* sm_task_ids;
  const uint32_t* sm_offsets;
  uint32_t* sm_cursors;
  uint32_t task_count;
  uint32_t resident_ctas;
  uint32_t sm_count;
  Dim3u logical_grid;
};

struct DeviceTrace {
  uint32_t* block_to_sm;
  uint32_t task_count;
};

struct LaunchConfig {
  dim3 persistent_grid;
  dim3 block_dim;
  uint32_t shared_memory_bytes;
};

__host__ __device__ inline uint32_t volume(Dim3u d) {
  return d.x * d.y * d.z;
}

__host__ __device__ inline uint32_t linearize(Dim3u grid, Dim3u block) {
  return block.x + block.y * grid.x + block.z * grid.x * grid.y;
}

__host__ __device__ inline Dim3u delinearize(Dim3u grid, uint32_t linear) {
  const uint32_t xy = grid.x * grid.y;
  const uint32_t z = linear / xy;
  const uint32_t rem = linear - z * xy;
  const uint32_t y = rem / grid.x;
  const uint32_t x = rem - y * grid.x;
  return Dim3u(x, y, z);
}

class HostSchedule {
 public:
  HostSchedule() : logical_grid_(1, 1, 1), resident_ctas_(0), sm_count_(0) {}

  HostSchedule(Dim3u logical_grid, uint32_t resident_ctas,
               uint32_t sm_count, std::vector<RTask> tasks,
               std::vector<uint32_t> sm_task_ids,
               std::vector<uint32_t> sm_offsets)
      : logical_grid_(logical_grid),
        resident_ctas_(resident_ctas),
        sm_count_(sm_count),
        tasks_(tasks),
        sm_task_ids_(sm_task_ids),
        sm_offsets_(sm_offsets) {}

  const std::vector<RTask>& tasks() const { return tasks_; }
  const std::vector<uint32_t>& sm_task_ids() const { return sm_task_ids_; }
  const std::vector<uint32_t>& sm_offsets() const { return sm_offsets_; }
  Dim3u logical_grid() const { return logical_grid_; }
  uint32_t task_count() const { return static_cast<uint32_t>(tasks_.size()); }
  uint32_t resident_ctas() const { return resident_ctas_; }
  uint32_t sm_count() const { return sm_count_; }

  LaunchConfig launch_config(dim3 block_dim,
                             uint32_t shared_memory_bytes = 0) const {
    return {dim3(resident_ctas_, 1, 1), block_dim, shared_memory_bytes};
  }

 private:
  Dim3u logical_grid_;
  uint32_t resident_ctas_;
  uint32_t sm_count_;
  std::vector<RTask> tasks_;
  std::vector<uint32_t> sm_task_ids_;
  std::vector<uint32_t> sm_offsets_;
};

struct RoundRobinSM {
  __host__ uint32_t operator()(uint32_t linear_block, Dim3u, uint32_t sm_count)
      const {
    return sm_count == 0 ? 0 : linear_block % sm_count;
  }
};

struct BlockedSM {
  uint32_t block_run;

  explicit BlockedSM(uint32_t run = 1) : block_run(run == 0 ? 1 : run) {}

  __host__ uint32_t operator()(uint32_t linear_block, Dim3u, uint32_t sm_count)
      const {
    return sm_count == 0 ? 0 : (linear_block / block_run) % sm_count;
  }
};

struct IdentityBlockOrder {
  __host__ uint32_t operator()(uint32_t logical_rank, Dim3u) const {
    return logical_rank;
  }
};

struct ColumnMajorBlockOrder {
  __host__ uint32_t operator()(uint32_t logical_rank, Dim3u grid) const {
    const uint32_t xy = grid.x * grid.y;
    const uint32_t z = logical_rank / xy;
    const uint32_t rem = logical_rank - z * xy;
    const uint32_t x = rem / grid.y;
    const uint32_t y = rem - x * grid.y;
    return linearize(grid, Dim3u(x, y, z));
  }
};

struct ExplicitBlockOrder {
  const uint32_t* order;
  uint32_t size;

  ExplicitBlockOrder(const uint32_t* order_, uint32_t size_)
      : order(order_), size(size_) {}

  __host__ uint32_t operator()(uint32_t logical_rank, Dim3u) const {
    return logical_rank < size ? order[logical_rank] : logical_rank;
  }
};

struct ExplicitSMMap {
  const uint32_t* block_to_sm;
  uint32_t size;

  ExplicitSMMap(const uint32_t* block_to_sm_, uint32_t size_)
      : block_to_sm(block_to_sm_), size(size_) {}

  __host__ uint32_t operator()(uint32_t linear_block, Dim3u, uint32_t sm_count)
      const {
    if (linear_block >= size) {
      return sm_count == 0 ? 0 : linear_block % sm_count;
    }
    return block_to_sm[linear_block];
  }
};

template <typename BlockOrder = IdentityBlockOrder,
          typename SMMapper = RoundRobinSM>
HostSchedule make_schedule(Dim3u logical_grid, uint32_t sm_count,
                           uint32_t ctas_per_sm = 1,
                           BlockOrder order = BlockOrder(),
                           SMMapper mapper = SMMapper()) {
  const uint32_t total_blocks = volume(logical_grid);
  const uint32_t resident_ctas = sm_count * ctas_per_sm;
  std::vector<RTask> tasks;
  tasks.reserve(total_blocks);
  std::vector<uint32_t> sm_counts(sm_count + 1, 0);

  for (uint32_t rank = 0; rank < total_blocks; ++rank) {
    const uint32_t linear_block = order(rank, logical_grid);
    const uint32_t sm = mapper(linear_block, logical_grid, sm_count);
    tasks.push_back({delinearize(logical_grid, linear_block), linear_block, sm});
    if (sm < sm_count) {
      sm_counts[sm + 1]++;
    }
  }

  std::vector<uint32_t> sm_offsets(sm_count + 1, 0);
  for (uint32_t sm = 0; sm < sm_count; ++sm) {
    sm_offsets[sm + 1] = sm_offsets[sm] + sm_counts[sm + 1];
  }

  std::vector<uint32_t> sm_task_ids(total_blocks, 0);
  std::vector<uint32_t> cursor = sm_offsets;
  for (uint32_t task_id = 0; task_id < total_blocks; ++task_id) {
    const uint32_t sm = tasks[task_id].sm;
    if (sm < sm_count) {
      sm_task_ids[cursor[sm]++] = task_id;
    }
  }

  return HostSchedule(logical_grid, resident_ctas, sm_count, tasks, sm_task_ids,
                      sm_offsets);
}

inline HostSchedule make_explicit_schedule(Dim3u logical_grid, uint32_t sm_count,
                                           const std::vector<uint32_t>& block_to_sm,
                                           uint32_t ctas_per_sm = 1) {
  return make_schedule(logical_grid, sm_count, ctas_per_sm,
                       IdentityBlockOrder(),
                       ExplicitSMMap(block_to_sm.data(),
                                     static_cast<uint32_t>(block_to_sm.size())));
}

class ScheduleBuilder {
 public:
  explicit ScheduleBuilder(Dim3u logical_grid)
      : logical_grid_(logical_grid), sm_count_(0), ctas_per_sm_(1) {}

  ScheduleBuilder& sms(uint32_t sm_count) {
    sm_count_ = sm_count;
    return *this;
  }

  ScheduleBuilder& ctas_per_sm(uint32_t ctas_per_sm) {
    ctas_per_sm_ = ctas_per_sm == 0 ? 1 : ctas_per_sm;
    return *this;
  }

  HostSchedule round_robin() const {
    return make_schedule(logical_grid_, sm_count_, ctas_per_sm_,
                         IdentityBlockOrder(), RoundRobinSM());
  }

  HostSchedule blocked(uint32_t block_run) const {
    return make_schedule(logical_grid_, sm_count_, ctas_per_sm_,
                         IdentityBlockOrder(), BlockedSM(block_run));
  }

  HostSchedule column_major_round_robin() const {
    return make_schedule(logical_grid_, sm_count_, ctas_per_sm_,
                         ColumnMajorBlockOrder(), RoundRobinSM());
  }

  HostSchedule explicit_sm(const std::vector<uint32_t>& block_to_sm) const {
    return make_explicit_schedule(logical_grid_, sm_count_, block_to_sm,
                                  ctas_per_sm_);
  }

  template <typename BlockOrder, typename SMMapper>
  HostSchedule custom(BlockOrder order, SMMapper mapper) const {
    return make_schedule(logical_grid_, sm_count_, ctas_per_sm_, order, mapper);
  }

 private:
  Dim3u logical_grid_;
  uint32_t sm_count_;
  uint32_t ctas_per_sm_;
};

class DeviceScheduleBuffer {
 public:
  DeviceScheduleBuffer()
      : tasks_(nullptr),
        sm_task_ids_(nullptr),
        sm_offsets_(nullptr),
        sm_cursors_(nullptr),
        task_count_(0),
        resident_ctas_(0),
        sm_count_(0) {}
  DeviceScheduleBuffer(const DeviceScheduleBuffer&) = delete;
  DeviceScheduleBuffer& operator=(const DeviceScheduleBuffer&) = delete;

  DeviceScheduleBuffer(DeviceScheduleBuffer&& other) noexcept
      : tasks_(other.tasks_),
        sm_task_ids_(other.sm_task_ids_),
        sm_offsets_(other.sm_offsets_),
        sm_cursors_(other.sm_cursors_),
        task_count_(other.task_count_),
        resident_ctas_(other.resident_ctas_),
        sm_count_(other.sm_count_),
        logical_grid_(other.logical_grid_) {
    other.tasks_ = nullptr;
    other.sm_task_ids_ = nullptr;
    other.sm_offsets_ = nullptr;
    other.sm_cursors_ = nullptr;
    other.task_count_ = 0;
    other.resident_ctas_ = 0;
    other.sm_count_ = 0;
  }

  DeviceScheduleBuffer& operator=(DeviceScheduleBuffer&& other) noexcept {
    if (this != &other) {
      release();
      tasks_ = other.tasks_;
      sm_task_ids_ = other.sm_task_ids_;
      sm_offsets_ = other.sm_offsets_;
      sm_cursors_ = other.sm_cursors_;
      task_count_ = other.task_count_;
      resident_ctas_ = other.resident_ctas_;
      sm_count_ = other.sm_count_;
      logical_grid_ = other.logical_grid_;
      other.tasks_ = nullptr;
      other.sm_task_ids_ = nullptr;
      other.sm_offsets_ = nullptr;
      other.sm_cursors_ = nullptr;
      other.task_count_ = 0;
      other.resident_ctas_ = 0;
      other.sm_count_ = 0;
    }
    return *this;
  }

  ~DeviceScheduleBuffer() { release(); }

  cudaError_t upload(const HostSchedule& schedule) {
    release();
    task_count_ = schedule.task_count();
    resident_ctas_ = schedule.resident_ctas();
    sm_count_ = schedule.sm_count();
    logical_grid_ = schedule.logical_grid();

    if (task_count_ == 0 || sm_count_ == 0) {
      return cudaSuccess;
    }

    cudaError_t err = cudaMalloc(&tasks_, sizeof(RTask) * task_count_);
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMalloc(&sm_task_ids_, sizeof(uint32_t) * task_count_);
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMalloc(&sm_offsets_, sizeof(uint32_t) * (sm_count_ + 1));
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMalloc(&sm_cursors_, sizeof(uint32_t) * sm_count_);
    if (err != cudaSuccess) {
      release();
      return err;
    }

    err = cudaMemcpy(tasks_, schedule.tasks().data(), sizeof(RTask) * task_count_,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMemcpy(sm_task_ids_, schedule.sm_task_ids().data(),
                     sizeof(uint32_t) * task_count_, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      release();
      return err;
    }
    err = cudaMemcpy(sm_offsets_, schedule.sm_offsets().data(),
                     sizeof(uint32_t) * (sm_count_ + 1), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      release();
      return err;
    }
    return cudaMemset(sm_cursors_, 0, sizeof(uint32_t) * sm_count_);
  }

  cudaError_t reset_cursors(cudaStream_t stream = 0) const {
    if (sm_cursors_ == nullptr || sm_count_ == 0) {
      return cudaSuccess;
    }
    return cudaMemsetAsync(sm_cursors_, 0, sizeof(uint32_t) * sm_count_, stream);
  }

  DeviceSchedule device_view() const {
    return {tasks_,      sm_task_ids_, sm_offsets_, sm_cursors_,
            task_count_, resident_ctas_, sm_count_, logical_grid_};
  }

  const RTask* data() const { return tasks_; }
  uint32_t size() const { return task_count_; }

 private:
  void release() {
    if (tasks_ != nullptr) {
      cudaFree(tasks_);
      tasks_ = nullptr;
    }
    if (sm_task_ids_ != nullptr) {
      cudaFree(sm_task_ids_);
      sm_task_ids_ = nullptr;
    }
    if (sm_offsets_ != nullptr) {
      cudaFree(sm_offsets_);
      sm_offsets_ = nullptr;
    }
    if (sm_cursors_ != nullptr) {
      cudaFree(sm_cursors_);
      sm_cursors_ = nullptr;
    }
    task_count_ = 0;
    resident_ctas_ = 0;
    sm_count_ = 0;
  }

  RTask* tasks_;
  uint32_t* sm_task_ids_;
  uint32_t* sm_offsets_;
  uint32_t* sm_cursors_;
  uint32_t task_count_;
  uint32_t resident_ctas_;
  uint32_t sm_count_;
  Dim3u logical_grid_;
};

class PreparedSchedule {
 public:
  PreparedSchedule() = default;

  cudaError_t upload(HostSchedule schedule) {
    host_ = schedule;
    return device_.upload(host_);
  }

  cudaError_t reset_cursors(cudaStream_t stream = 0) const {
    return device_.reset_cursors(stream);
  }

  DeviceSchedule device_view() const { return device_.device_view(); }
  LaunchConfig launch_config(dim3 block_dim,
                             uint32_t shared_memory_bytes = 0) const {
    return host_.launch_config(block_dim, shared_memory_bytes);
  }

  const HostSchedule& host() const { return host_; }
  uint32_t task_count() const { return host_.task_count(); }

 private:
  HostSchedule host_;
  DeviceScheduleBuffer device_;
};

class TraceBuffer {
 public:
  TraceBuffer() : block_to_sm_(nullptr), task_count_(0) {}
  TraceBuffer(const TraceBuffer&) = delete;
  TraceBuffer& operator=(const TraceBuffer&) = delete;

  TraceBuffer(TraceBuffer&& other) noexcept
      : block_to_sm_(other.block_to_sm_), task_count_(other.task_count_) {
    other.block_to_sm_ = nullptr;
    other.task_count_ = 0;
  }

  TraceBuffer& operator=(TraceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      block_to_sm_ = other.block_to_sm_;
      task_count_ = other.task_count_;
      other.block_to_sm_ = nullptr;
      other.task_count_ = 0;
    }
    return *this;
  }

  ~TraceBuffer() { release(); }

  cudaError_t allocate(uint32_t task_count) {
    release();
    task_count_ = task_count;
    if (task_count_ == 0) {
      return cudaSuccess;
    }
    cudaError_t err = cudaMalloc(&block_to_sm_, sizeof(uint32_t) * task_count_);
    if (err != cudaSuccess) {
      release();
      return err;
    }
    return reset();
  }

  cudaError_t reset(cudaStream_t stream = 0) const {
    if (block_to_sm_ == nullptr || task_count_ == 0) {
      return cudaSuccess;
    }
    return cudaMemsetAsync(block_to_sm_, 0xff, sizeof(uint32_t) * task_count_, stream);
  }

  cudaError_t download(std::vector<uint32_t>* out) const {
    out->assign(task_count_, kInvalidSM);
    if (block_to_sm_ == nullptr || task_count_ == 0) {
      return cudaSuccess;
    }
    return cudaMemcpy(out->data(), block_to_sm_, sizeof(uint32_t) * task_count_,
                      cudaMemcpyDeviceToHost);
  }

  DeviceTrace device_view() const { return {block_to_sm_, task_count_}; }
  uint32_t size() const { return task_count_; }

 private:
  void release() {
    if (block_to_sm_ != nullptr) {
      cudaFree(block_to_sm_);
      block_to_sm_ = nullptr;
    }
    task_count_ = 0;
  }

  uint32_t* block_to_sm_;
  uint32_t task_count_;
};

class PreparedTrace {
 public:
  PreparedTrace() = default;

  cudaError_t allocate(uint32_t task_count) { return trace_.allocate(task_count); }

  cudaError_t reset(cudaStream_t stream = 0) const {
    return trace_.reset(stream);
  }

  DeviceTrace device_view() const { return trace_.device_view(); }

  cudaError_t download(std::vector<uint32_t>* out) const {
    return trace_.download(out);
  }

  uint32_t size() const { return trace_.size(); }

 private:
  TraceBuffer trace_;
};

__device__ inline void record_trace(DeviceTrace trace, const RTask& task,
                                    uint32_t sm) {
  if (trace.block_to_sm != nullptr && task.linear_block < trace.task_count) {
    trace.block_to_sm[task.linear_block] = sm;
  }
}

__device__ inline uint32_t persistent_cta_id() {
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

__device__ inline uint32_t current_sm_id() {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

__device__ inline bool is_block_leader() {
  return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0;
}

template <typename Body>
__device__ void run_persistent(DeviceSchedule schedule, Body body) {
  const uint32_t cta = persistent_cta_id();
  const uint32_t stride = schedule.resident_ctas == 0 ? gridDim.x : schedule.resident_ctas;

  for (uint32_t task_id = cta; task_id < schedule.task_count; task_id += stride) {
    RTask task = schedule.tasks[task_id];
    body(task);
  }
}

template <typename Body>
__device__ void run_persistent_sm_affine(DeviceSchedule schedule, Body body) {
  const uint32_t sm = current_sm_id();
  if (sm >= schedule.sm_count || schedule.sm_offsets == nullptr ||
      schedule.sm_task_ids == nullptr || schedule.sm_cursors == nullptr) {
    return;
  }

  const uint32_t begin = schedule.sm_offsets[sm];
  const uint32_t end = schedule.sm_offsets[sm + 1];
  __shared__ uint32_t claimed_pos;
  __shared__ uint32_t task_block_x;
  __shared__ uint32_t task_block_y;
  __shared__ uint32_t task_block_z;
  __shared__ uint32_t task_linear_block;
  __shared__ uint32_t task_sm;
  while (true) {
    if (is_block_leader()) {
      const uint32_t local = atomicAdd(&schedule.sm_cursors[sm], 1);
      claimed_pos = begin + local;
      if (claimed_pos < end) {
        const RTask loaded = schedule.tasks[schedule.sm_task_ids[claimed_pos]];
        task_block_x = loaded.block.x;
        task_block_y = loaded.block.y;
        task_block_z = loaded.block.z;
        task_linear_block = loaded.linear_block;
        task_sm = loaded.sm;
      }
    }
    __syncthreads();
    if (claimed_pos >= end) {
      break;
    }
    const RTask task = {Dim3u(task_block_x, task_block_y, task_block_z),
                        task_linear_block, task_sm};
    body(task);
    __syncthreads();
  }
}

}  // namespace block_sche
