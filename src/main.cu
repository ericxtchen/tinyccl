#include "../include/cpu_device.hpp"
#include "../include/gpu_device.cuh"
#include "../include/ring_allreduce.cuh"
#include "../include/socket_transport.hpp"
#include <array>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  bool use_gpu = false;
  std::vector<std::string> positional_args;

  // Parse command-line flags and positional arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--gpu") {
      use_gpu = true;
    } else {
      positional_args.push_back(arg);
    }
  }

  if (positional_args.size() < 2) {
    std::cerr << "Usage: " << argv[0] << " [--gpu] <rank> <world_size>\n";
    return 1;
  }

  int rank = std::stoi(positional_args[0]);
  int world_size = std::stoi(positional_args[1]);
  constexpr int BASE_PORT = 8000;

  SocketTransport t(BASE_PORT, rank, world_size);

  // Dispatch based on the --gpu flag
  if (use_gpu) {
    std::cout << "Rank " << rank << " running on GPU...\n";

    std::array<float, 5> host_arr = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::size_t bytes = host_arr.size() * sizeof(float);

    // Allocate GPU memory (CUDA kernels need device pointers)
    float *d_arr = nullptr;
    cudaMalloc(&d_arr, bytes);
    cudaMemcpy(d_arr, host_arr.data(), bytes, cudaMemcpyHostToDevice);

    GpuDevice<float> d{};
    Ring_AllReduce<float, SocketTransport, GpuDevice<float>> allreduce(
        t, d, rank, world_size);

    allreduce.execute(d_arr, host_arr.size());

    // Copy result back to CPU to print
    cudaMemcpy(host_arr.data(), d_arr, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    std::cout << "Rank " << rank << " GPU result: ";
    for (float v : host_arr)
      std::cout << v << " ";
    std::cout << "\n";

  } else {
    std::cout << "Rank " << rank << " running on CPU...\n";

    std::array<float, 5> arr = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    CpuDevice<float> d{};
    Ring_AllReduce<float, SocketTransport, CpuDevice<float>> allreduce(
        t, d, rank, world_size);

    allreduce.execute(arr.data(), arr.size());

    std::cout << "Rank " << rank << " CPU result: ";
    for (float v : arr)
      std::cout << v << " ";
    std::cout << "\n";
  }

  return 0;
}