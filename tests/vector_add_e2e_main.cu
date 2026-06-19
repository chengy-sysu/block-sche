#include <block_sche/block_sche.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

__global__ void vector_add_persistent(block_sche::DeviceSchedule schedule,
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

  auto host_schedule = block_sche::make_schedule(
      block_sche::Dim3u(static_cast<uint32_t>(logical_blocks), 1, 1),
      static_cast<uint32_t>(sm_count), 1);
  block_sche::PreparedSchedule schedule;
  if (!check(schedule.upload(host_schedule), "schedule upload")) {
    return 1;
  }

  const auto launch = schedule.launch_config(dim3(threads, 1, 1));
  vector_add_persistent<<<launch.persistent_grid, launch.block_dim>>>(
      schedule.device_view(), d_out, d_a, d_b, n);
  if (!check(cudaGetLastError(), "vector_add_persistent launch") ||
      !check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    return 1;
  }

  if (!check(cudaMemcpy(h_out.data(), d_out, sizeof(float) * n, cudaMemcpyDeviceToHost),
             "cudaMemcpy h_out")) {
    return 1;
  }

  for (int i = 0; i < n; ++i) {
    const float expected = h_a[i] + h_b[i];
    if (std::fabs(h_out[i] - expected) > 1.0e-5f) {
      std::fprintf(stderr, "mismatch at %d: got %f expected %f\n", i, h_out[i],
                   expected);
      return 1;
    }
  }

  cudaFree(d_out);
  cudaFree(d_b);
  cudaFree(d_a);
  return 0;
}
