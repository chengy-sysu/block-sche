#include <block_sche/block_sche.cuh>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

__global__ void sgemm_naive_f32_kernel_persistent(
    block_sche::DeviceSchedule schedule, float* a, float* b, float* c, int M,
    int N, int K);

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

  constexpr int m = 37;
  constexpr int n = 41;
  constexpr int k = 29;
  constexpr int block_x = 16;
  constexpr int block_y = 8;
  const int grid_x = (n + block_x - 1) / block_x;
  const int grid_y = (m + block_y - 1) / block_y;

  std::vector<float> h_a(m * k);
  std::vector<float> h_b(k * n);
  std::vector<float> h_c(m * n, -1.0f);
  for (int i = 0; i < m * k; ++i) {
    h_a[i] = static_cast<float>((i % 17) - 8) * 0.125f;
  }
  for (int i = 0; i < k * n; ++i) {
    h_b[i] = static_cast<float>((i % 19) - 6) * 0.0625f;
  }

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  if (!check(cudaMalloc(&d_a, sizeof(float) * h_a.size()), "cudaMalloc d_a") ||
      !check(cudaMalloc(&d_b, sizeof(float) * h_b.size()), "cudaMalloc d_b") ||
      !check(cudaMalloc(&d_c, sizeof(float) * h_c.size()), "cudaMalloc d_c")) {
    return 1;
  }
  if (!check(cudaMemcpy(d_a, h_a.data(), sizeof(float) * h_a.size(),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy d_a") ||
      !check(cudaMemcpy(d_b, h_b.data(), sizeof(float) * h_b.size(),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy d_b") ||
      !check(cudaMemset(d_c, 0, sizeof(float) * h_c.size()), "cudaMemset d_c")) {
    return 1;
  }

  auto host_schedule =
      block_sche::ScheduleBuilder(block_sche::Dim3u(grid_x, grid_y, 1))
          .sms(static_cast<uint32_t>(sm_count))
          .ctas_per_sm(1)
          .column_major_round_robin();
  block_sche::PreparedSchedule schedule;
  if (!check(schedule.upload(host_schedule), "schedule upload")) {
    return 1;
  }

  const auto launch = schedule.launch_config(dim3(block_x, block_y, 1));
  sgemm_naive_f32_kernel_persistent<<<launch.persistent_grid,
                                      launch.block_dim>>>(
      schedule.device_view(), d_a, d_b, d_c, m, n, k);
  if (!check(cudaGetLastError(), "sgemm persistent launch") ||
      !check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    return 1;
  }

  if (!check(cudaMemcpy(h_c.data(), d_c, sizeof(float) * h_c.size(),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy h_c")) {
    return 1;
  }

  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float expected = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        expected += h_a[row * k + kk] * h_b[kk * n + col];
      }
      const float got = h_c[row * n + col];
      if (std::fabs(got - expected) > 1.0e-4f) {
        std::fprintf(stderr,
                     "mismatch at row=%d col=%d: got %f expected %f\n",
                     row, col, got, expected);
        return 1;
      }
    }
  }

  cudaFree(d_c);
  cudaFree(d_b);
  cudaFree(d_a);
  return 0;
}
