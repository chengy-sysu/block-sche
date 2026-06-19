#include <block_sche/block_sche.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

__global__ void vector_add_persistent(block_sche::DeviceSchedule schedule,
                                      block_sche::DeviceTrace trace,
                                      float* out, const float* a,
                                      const float* b, int n);

static bool check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    return false;
  }
  return true;
}

int main() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "no CUDA device available\n");
    return 77;
  }

  int sm_count = 0;
  if (!check(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0),
             "cudaDeviceGetAttribute")) {
    return 1;
  }

  constexpr int n = 1 << 15;
  constexpr int threads = 128;
  const int logical_blocks = (n + threads - 1) / threads;

  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<float> h_out(n, -1.0f);
  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i % 17);
    h_b[i] = static_cast<float>(i % 23) * 0.5f;
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_out = nullptr;
  if (!check(cudaMalloc(&d_a, sizeof(float) * n), "cudaMalloc d_a") ||
      !check(cudaMalloc(&d_b, sizeof(float) * n), "cudaMalloc d_b") ||
      !check(cudaMalloc(&d_out, sizeof(float) * n), "cudaMalloc d_out")) {
    return 1;
  }

  if (!check(cudaMemcpy(d_a, h_a.data(), sizeof(float) * n, cudaMemcpyHostToDevice),
             "cudaMemcpy d_a") ||
      !check(cudaMemcpy(d_b, h_b.data(), sizeof(float) * n, cudaMemcpyHostToDevice),
             "cudaMemcpy d_b") ||
      !check(cudaMemset(d_out, 0, sizeof(float) * n), "cudaMemset d_out")) {
    return 1;
  }

  auto host_schedule =
      block_sche::ScheduleBuilder(
          block_sche::Dim3u(static_cast<uint32_t>(logical_blocks), 1, 1))
          .sms(static_cast<uint32_t>(sm_count))
          .ctas_per_sm(1)
          .round_robin();

  block_sche::PreparedSchedule schedule;
  block_sche::PreparedTrace trace;
  if (!check(schedule.upload(host_schedule), "schedule upload") ||
      !check(trace.allocate(schedule.task_count()), "trace allocate")) {
    return 1;
  }

  const auto launch = schedule.launch_config(dim3(threads, 1, 1));
  vector_add_persistent<<<launch.persistent_grid, launch.block_dim>>>(
      schedule.device_view(), trace.device_view(), d_out, d_a, d_b, n);
  if (!check(cudaGetLastError(), "vector_add_persistent launch") ||
      !check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    return 1;
  }

  if (!check(cudaMemcpy(h_out.data(), d_out, sizeof(float) * n, cudaMemcpyDeviceToHost),
             "cudaMemcpy h_out")) {
    return 1;
  }

  std::vector<uint32_t> observed_trace;
  if (!check(trace.download(&observed_trace), "trace download")) {
    return 1;
  }

  std::vector<uint32_t> observed_sm_hist(static_cast<size_t>(sm_count), 0);
  uint32_t missing_tasks = 0;
  uint32_t mismatched_tasks = 0;
  for (const auto& task : host_schedule.tasks()) {
    const uint32_t observed_sm = observed_trace[task.linear_block];
    if (observed_sm == block_sche::kInvalidSM) {
      missing_tasks++;
    } else if (observed_sm < static_cast<uint32_t>(sm_count)) {
      observed_sm_hist[observed_sm]++;
    }
    if (observed_sm != task.sm) {
      mismatched_tasks++;
    }
  }

  for (int i = 0; i < n; ++i) {
    const float expected = h_a[i] + h_b[i];
    if (std::fabs(h_out[i] - expected) > 1.0e-5f) {
      std::fprintf(stderr, "mismatch at %d: got %f expected %f\n", i, h_out[i],
                   expected);
      std::fprintf(stderr, "trace summary: missing_tasks=%u mismatched_tasks=%u sm_count=%d\n",
                   missing_tasks, mismatched_tasks, sm_count);
      std::fprintf(stderr, "observed SM histogram:");
      for (int sm = 0; sm < std::min(sm_count, 32); ++sm) {
        std::fprintf(stderr, " %d:%u", sm, observed_sm_hist[sm]);
      }
      if (sm_count > 32) {
        std::fprintf(stderr, " ...");
      }
      std::fprintf(stderr, "\nfirst missing/mismatched tasks:");
      uint32_t printed = 0;
      for (const auto& task : host_schedule.tasks()) {
        const uint32_t observed_sm = observed_trace[task.linear_block];
        if (observed_sm != task.sm) {
          std::fprintf(stderr, " block=%u requested=%u observed=%u",
                       task.linear_block, task.sm, observed_sm);
          if (++printed == 12) {
            break;
          }
        }
      }
      std::fprintf(stderr, "\n");
      return 1;
    }
  }

  for (const auto& task : host_schedule.tasks()) {
    const uint32_t observed_sm = observed_trace[task.linear_block];
    if (observed_sm != task.sm) {
      std::fprintf(stderr,
                   "SM mismatch for block %u: requested %u observed %u\n",
                   task.linear_block, task.sm, observed_sm);
      std::fprintf(stderr, "trace summary: missing_tasks=%u mismatched_tasks=%u sm_count=%d\n",
                   missing_tasks, mismatched_tasks, sm_count);
      return 1;
    }
  }

  cudaFree(d_out);
  cudaFree(d_b);
  cudaFree(d_a);
  return 0;
}
