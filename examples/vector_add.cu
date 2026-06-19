#include <cuda_runtime.h>

__global__ void vector_add(float* out, const float* a, const float* b, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = tid; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}
