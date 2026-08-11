// ring_allreduce = reduce scatter + allgather
// do n - 1 steps of reduce scatter
// we receive some chunk -> run our reduce function on it -> send it to neighbor
// what if we had a thread that is in a worker loop waiting to receive data? how
// does this fit with our current recv method signature? have to open a socket
// connection on BASE_PORT + rank_ and listen from there once we get all data
// from recv, run the reduce function and then send do we even need multiple
// threads? we need to make sure there is no deadlock
#include "../include/ring_allreduce.hpp"

template <typename ValueType, TransportPolicy Transport,
          DevicePolicy<ValueType> Device>
void Ring_AllReduce<ValueType, Transport, Device>::execute(
    ValueType *data, std::size_t total_elements) {
  // this is the amount of elements each GPU gets
  std::size_t chunk_size = total_elements / this->world_size_;
  std::size_t chunk_size_bytes = chunk_size * sizeof(ValueType);

  int send_rank = (this->rank_ + 1) % this->world_size_;
  int recv_rank = (this->rank_ - 1 + this->world_size_) % this->world_size_;

  // Reduce-Scatter phase of N-1 steps
  for (int step = 0; step < this->world_size_ - 1; ++step) {
    t_.recv();
  }
};