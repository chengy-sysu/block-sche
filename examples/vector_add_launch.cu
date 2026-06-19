#include <block_sche/block_sche.cuh>

#include <cuda_runtime.h>

__global__ void vector_add_persistent(block_sche::DeviceSchedule schedule,
                                      float* out, const float* a,
                                      const float* b, int n);

struct EvenOddSM {
  __host__ uint32_t operator()(uint32_t linear_block, block_sche::Dim3u,
                               uint32_t sm_count) const {
    if (sm_count == 0) {
      return 0;
    }
    const uint32_t half = (sm_count + 1) / 2;
    return (linear_block % 2) * half + (linear_block / 2) % half;
  }
};

cudaError_t launch_vector_add(float* out, const float* a, const float* b, int n,
                              cudaStream_t stream) {
  int sm_count = 0;
  cudaError_t err =
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
  if (err != cudaSuccess) {
    return err;
  }

  const dim3 block_dim(256, 1, 1);
  const block_sche::Dim3u logical_grid((n + block_dim.x - 1) / block_dim.x, 1, 1);
  const uint32_t ctas_per_sm = 1;

  auto host_schedule = block_sche::ScheduleBuilder(logical_grid)
                           .sms(static_cast<uint32_t>(sm_count))
                           .ctas_per_sm(ctas_per_sm)
                           .custom(block_sche::IdentityBlockOrder(), EvenOddSM());

  block_sche::PreparedSchedule schedule;
  err = schedule.upload(host_schedule);
  if (err != cudaSuccess) {
    return err;
  }

  const auto launch = schedule.launch_config(block_dim);
  vector_add_persistent<<<launch.persistent_grid, launch.block_dim,
                          launch.shared_memory_bytes, stream>>>(
      schedule.device_view(), out, a, b, n);
  return cudaGetLastError();
}
