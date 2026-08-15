// ring_allreduce = reduce scatter + allgather
// do n - 1 steps of reduce scatter
// we receive some chunk -> run our reduce function on it -> send it to neighbor
// what if we had a thread that is in a worker loop waiting to receive data? how
// does this fit with our current recv method signature? have to open a socket
// connection on BASE_PORT + rank_ and listen from there once we get all data
// from recv, run the reduce function and then send do we even need multiple
// threads? we need to make sure there is no deadlock

#ifndef RING_ALLREDUCE_H
#define RING_ALLREDUCE_H

#include "../include/device.hpp"
#include "../include/transport.hpp"
#include <cuda_runtime.h>
#include <vector>

template <typename ValueType, TransportPolicy Transport,
          DevicePolicy<ValueType> Device>
class Ring_AllReduce {
public:
  Ring_AllReduce(Transport &t, Device &d, int r, int N)
      : t_(t), d_(d), rank_(r), world_size_(N) {}

  // data is a pointer in GPU memory
  void execute(ValueType *data, std::size_t total_elements) {
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
        // if the device is a gpu, we want to copy all data from send_ptr +
        // chunk_size_bytes onwards into a cpu held memory buffer and then use
        // that to pass to t_.send()
        // same for the recv
        // then we need a buffer in the gpu that takes whatever we recvd to it
        // can be passed to reduce_add
        cudaMemcpy(send_cpu_buffer, send_ptr, chunk_size_bytes,
                   cudaMemcpyDeviceToHost);
        t_.send(send_cpu_buffer, chunk_size_bytes, send_rank);
        t_.recv(recv_cpu_buffer, chunk_size_bytes, recv_rank);
        cudaMemcpy(gpu_buffer, recv_cpu_buffer, chunk_size_bytes,
                   cudaMemcpyHostToDevice);

        d_.reduce_add(gpu_buffer, recv_ptr, chunk_size);
      } else {

        std::vector<ValueType> recv_buf(chunk_size);

        t_.send(send_ptr, chunk_size_bytes, send_rank);
        t_.recv(recv_buf.data(), chunk_size_bytes, recv_rank);

        d_.reduce_add(recv_buf.data(), recv_ptr, chunk_size);
      }
    }

    // All-Gather phase of N-1 steps
    for (int step = 0; step < world_size_ - 1; ++step) {
      int send_chunk = (rank_ - step + 1 + world_size_) % world_size_;
      int recv_chunk = (rank_ - step + world_size_) % world_size_;

      ValueType *send_ptr = data + (send_chunk * chunk_size);
      ValueType *recv_ptr = data + (recv_chunk * chunk_size);

      if constexpr (Device::is_gpu) {
        cudaMemcpy(send_cpu_buffer, send_ptr, chunk_size_bytes,
                   cudaMemcpyDeviceToHost);

        t_.send(send_cpu_buffer, chunk_size_bytes, send_rank);
        t_.recv(recv_cpu_buffer, chunk_size_bytes, recv_rank);

        cudaMemcpy(recv_ptr, recv_cpu_buffer, chunk_size_bytes,
                   cudaMemcpyHostToDevice);

      } else {
        t_.send(send_ptr, chunk_size_bytes, send_rank);
        t_.recv(recv_ptr, chunk_size_bytes, recv_rank);
      }
    }

    if constexpr (Device::is_gpu) {
      cudaFreeHost(send_cpu_buffer);
      cudaFreeHost(recv_cpu_buffer);
      cudaFree(gpu_buffer);
    }
  }

private:
  Transport t_;
  Device d_;
  int rank_;
  int world_size_;
};

#endif