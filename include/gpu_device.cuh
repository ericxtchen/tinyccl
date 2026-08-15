#ifndef GPUDEVICE_H
#define GPUDEVICE_H

#include <cuda_runtime.h>

template <typename T>
__global__ void reduce_add_kernel(const T *src_buffer, T *dest_buffer,
                                  std::size_t count) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < count) {
    dest_buffer[id] += src_buffer[id];
  }
}

template <typename T> class GpuDevice {
public:
  static constexpr bool is_gpu = true;
  void reduce_add(const T *src_buffer, T *dest_buffer, std::size_t count) {
    if (count == 0)
      return;

    // Configure thread/block dimensions
    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    reduce_add_kernel<T><<<blocks, threads>>>(src_buffer, dest_buffer, count);

    // Ensure the kernel finishes before returning
    cudaDeviceSynchronize();
  }
};

#endif