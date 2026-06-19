#include <block_sche/block_sche.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

__global__ void elementwise_add_f32x4_kernel_persistent(
    block_sche::DeviceSchedule schedule, float* a, float* b, float* c, int N);

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

  constexpr int n = 1 << 16;
  constexpr int logical_threads = 256;
  constexpr int elements_per_thread = 4;
  constexpr int block_threads = logical_threads / elements_per_thread;
  const int logical_blocks = (n + logical_threads - 1) / logical_threads;

  std::vector<float> h_a(n);
  std::vector<float> h_b(n);
  std::vector<float> h_c(n, -1.0f);
  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>((i % 127) - 31) * 0.25f;
    h_b[i] = static_cast<float>((i % 53) + 7) * 0.5f;
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  if (!check(cudaMalloc(&d_a, sizeof(float) * n), "cudaMalloc d_a") ||
      !check(cudaMalloc(&d_b, sizeof(float) * n), "cudaMalloc d_b") ||
      !check(cudaMalloc(&d_c, sizeof(float) * n), "cudaMalloc d_c")) {
    return 1;
  }
  if (!check(cudaMemcpy(d_a, h_a.data(), sizeof(float) * n, cudaMemcpyHostToDevice),
             "cudaMemcpy d_a") ||
      !check(cudaMemcpy(d_b, h_b.data(), sizeof(float) * n, cudaMemcpyHostToDevice),
             "cudaMemcpy d_b") ||
      !check(cudaMemset(d_c, 0, sizeof(float) * n), "cudaMemset d_c")) {
    return 1;
  }

  auto host_schedule =
      block_sche::ScheduleBuilder(block_sche::Dim3u(logical_blocks, 1, 1))
          .sms(static_cast<uint32_t>(sm_count))
          .ctas_per_sm(1)
          .blocked(3);
  block_sche::PreparedSchedule schedule;
  if (!check(schedule.upload(host_schedule), "schedule upload")) {
    return 1;
  }

  const auto launch = schedule.launch_config(dim3(block_threads, 1, 1));
  elementwise_add_f32x4_kernel_persistent<<<launch.persistent_grid,
                                            launch.block_dim>>>(
      schedule.device_view(), d_a, d_b, d_c, n);
  if (!check(cudaGetLastError(), "elementwise persistent launch") ||
      !check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    return 1;
  }

  if (!check(cudaMemcpy(h_c.data(), d_c, sizeof(float) * n, cudaMemcpyDeviceToHost),
             "cudaMemcpy h_c")) {
    return 1;
  }
  for (int i = 0; i < n; ++i) {
    const float expected = h_a[i] + h_b[i];
    if (std::fabs(h_c[i] - expected) > 1.0e-5f) {
      std::fprintf(stderr, "mismatch at %d: got %f expected %f\n", i, h_c[i],
                   expected);
      return 1;
    }
  }

  cudaFree(d_c);
  cudaFree(d_b);
  cudaFree(d_a);
  return 0;
}
