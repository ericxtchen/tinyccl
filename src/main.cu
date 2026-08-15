#include "../include/cpu_device.hpp"
#include "../include/gpu_device.cuh"
#include "../include/ring_allreduce.cuh"
#include "../include/socket_transport.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
  bool use_gpu = false;
  std::size_t requested_size = 1000; // Default size
  std::vector<std::string> positional_args;

  // Parse command-line flags and positional arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--gpu") {
      use_gpu = true;
    } else if (arg == "--size" && i + 1 < argc) {
      requested_size = std::stoull(argv[++i]);
    } else {
      positional_args.push_back(arg);
    }
  }

  if (positional_args.size() < 2) {
    std::cerr << "Usage: " << argv[0]
              << " [--gpu] [--size N] <rank> <world_size>\n";
    return 1;
  }

  int rank = std::stoi(positional_args[0]);
  int world_size = std::stoi(positional_args[1]);
  constexpr int BASE_PORT = 8000;

  // Ensure size is a multiple of world_size for even chunk division
  if (requested_size % world_size != 0) {
    requested_size = ((requested_size / world_size) + 1) * world_size;
  }

  // Populate dynamic vector with test numerical data
  std::vector<float> host_arr(requested_size);
  for (std::size_t i = 0; i < requested_size; ++i) {
    host_arr[i] = static_cast<float>(i + 1);
  }

  SocketTransport t(BASE_PORT, rank, world_size);
  std::size_t bytes = host_arr.size() * sizeof(float);

  if (use_gpu) {
    std::cout << "Rank " << rank << " running on GPU with " << host_arr.size()
              << " elements...\n";

    float *d_arr = nullptr;
    cudaMalloc(&d_arr, bytes);
    cudaMemcpy(d_arr, host_arr.data(), bytes, cudaMemcpyHostToDevice);

    GpuDevice<float> d{};
    Ring_AllReduce<float, SocketTransport, GpuDevice<float>> allreduce(
        t, d, rank, world_size);

    allreduce.execute(d_arr, host_arr.size());

    cudaMemcpy(host_arr.data(), d_arr, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

  } else {
    std::cout << "Rank " << rank << " running on CPU with " << host_arr.size()
              << " elements...\n";

    CpuDevice<float> d{};
    Ring_AllReduce<float, SocketTransport, CpuDevice<float>> allreduce(
        t, d, rank, world_size);

    allreduce.execute(host_arr.data(), host_arr.size());
  }

  std::cout << "Rank " << rank
            << " execution complete. First element: " << host_arr[0]
            << ", Last element: " << host_arr.back() << "\n";

  return 0;
}