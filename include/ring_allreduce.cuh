// ring_allreduce = reduce scatter + allgather
#ifndef RING_ALLREDUCE_H
#define RING_ALLREDUCE_H

#include "../include/device.hpp"
#include "../include/transport.hpp"
#include <chrono>
#include <cuda_runtime.h>
#include <vector>

// Timing breakdown for one execute() call. staging_ms is 0 for CpuDevice
// (there's no host<->device copy on that path).
struct RingTiming {
  double total_ms = 0.0;
  double staging_ms = 0.0;  // cudaMemcpy device<->host bounce-buffer copies
  double transfer_ms = 0.0; // network sendrecv
  double compute_ms = 0.0;  // reduce_add
};

template <typename ValueType, TransportPolicy Transport,
          DevicePolicy<ValueType> Device>
class Ring_AllReduce {
public:
  Ring_AllReduce(Transport &t, Device &d, int r, int N)
      : t_(t), d_(d), rank_(r), world_size_(N) {}

  // data is a pointer in GPU memory
  RingTiming execute(ValueType *data, std::size_t total_elements) {
    using Clock = std::chrono::high_resolution_clock;
    auto ms = [](Clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };

    RingTiming timing;
    auto exec_start = Clock::now();

    // This is the amount of elements each GPU gets.
    std::size_t chunk_size = total_elements / this->world_size_;
    std::size_t chunk_size_bytes = chunk_size * sizeof(ValueType);

    int send_rank = (this->rank_ + 1) % this->world_size_;
    int recv_rank = (this->rank_ - 1 + this->world_size_) % this->world_size_;

    ValueType *send_cpu_buffer = nullptr;
    ValueType *recv_cpu_buffer = nullptr;
    ValueType *gpu_buffer = nullptr;

    if constexpr (Device::is_gpu) {
      cudaMallocHost((void **)&send_cpu_buffer, chunk_size_bytes);
      cudaMallocHost((void **)&recv_cpu_buffer, chunk_size_bytes);
      cudaMalloc((void **)&gpu_buffer, chunk_size_bytes);
    }

    // Reduce-Scatter phase of N-1 steps
    for (int step = 0; step < this->world_size_ - 1; ++step) {
      int send_chunk = (rank_ - step + world_size_) % world_size_;
      int recv_chunk = (rank_ - step - 1 + world_size_) % world_size_;

      ValueType *send_ptr = data + (send_chunk * chunk_size);
      ValueType *recv_ptr = data + (recv_chunk * chunk_size);

      if constexpr (Device::is_gpu) {
        auto t0 = Clock::now();
        cudaMemcpy(send_cpu_buffer, send_ptr, chunk_size_bytes,
                   cudaMemcpyDeviceToHost);
        auto t1 = Clock::now();

        // use sendrecv where send is is run on another thread to prevent
        // deadlock
        t_.sendrecv(send_cpu_buffer, chunk_size_bytes, send_rank,
                    recv_cpu_buffer, chunk_size_bytes, recv_rank);
        auto t2 = Clock::now();

        cudaMemcpy(gpu_buffer, recv_cpu_buffer, chunk_size_bytes,
                   cudaMemcpyHostToDevice);
        auto t3 = Clock::now();

        d_.reduce_add(gpu_buffer, recv_ptr, chunk_size);
        auto t4 = Clock::now();

        timing.staging_ms += ms(t1 - t0) + ms(t3 - t2);
        timing.transfer_ms += ms(t2 - t1);
        timing.compute_ms += ms(t4 - t3);
      } else {
        std::vector<ValueType> recv_buf(chunk_size);

        auto t0 = Clock::now();
        t_.sendrecv(send_ptr, chunk_size_bytes, send_rank, recv_buf.data(),
                    chunk_size_bytes, recv_rank);
        auto t1 = Clock::now();

        d_.reduce_add(recv_buf.data(), recv_ptr, chunk_size);
        auto t2 = Clock::now();

        timing.transfer_ms += ms(t1 - t0);
        timing.compute_ms += ms(t2 - t1);
      }
    }

    // All-Gather phase of N-1 steps
    for (int step = 0; step < world_size_ - 1; ++step) {
      int send_chunk = (rank_ - step + 1 + world_size_) % world_size_;
      int recv_chunk = (rank_ - step + world_size_) % world_size_;

      ValueType *send_ptr = data + (send_chunk * chunk_size);
      ValueType *recv_ptr = data + (recv_chunk * chunk_size);

      if constexpr (Device::is_gpu) {
        auto t0 = Clock::now();
        cudaMemcpy(send_cpu_buffer, send_ptr, chunk_size_bytes,
                   cudaMemcpyDeviceToHost);
        auto t1 = Clock::now();

        t_.sendrecv(send_cpu_buffer, chunk_size_bytes, send_rank,
                    recv_cpu_buffer, chunk_size_bytes, recv_rank);
        auto t2 = Clock::now();

        cudaMemcpy(recv_ptr, recv_cpu_buffer, chunk_size_bytes,
                   cudaMemcpyHostToDevice);
        auto t3 = Clock::now();

        timing.staging_ms += ms(t1 - t0) + ms(t3 - t2);
        timing.transfer_ms += ms(t2 - t1);
      } else {
        auto t0 = Clock::now();
        t_.sendrecv(send_ptr, chunk_size_bytes, send_rank, recv_ptr,
                    chunk_size_bytes, recv_rank);
        auto t1 = Clock::now();

        timing.transfer_ms += ms(t1 - t0);
      }
    }

    if constexpr (Device::is_gpu) {
      cudaFreeHost(send_cpu_buffer);
      cudaFreeHost(recv_cpu_buffer);
      cudaFree(gpu_buffer);
    }

    timing.total_ms = ms(Clock::now() - exec_start);
    return timing;
  }

private:
  Transport t_;
  Device d_;
  int rank_;
  int world_size_;
};

#endif
