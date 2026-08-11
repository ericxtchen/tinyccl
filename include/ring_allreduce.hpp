#ifndef RING_ALLREDUCE_H
#define RING_ALLREDUCE_H

#include "../include/device.hpp"
#include "../include/transport.hpp"

template <typename ValueType, TransportPolicy Transport,
          DevicePolicy<ValueType> Device>
class Ring_AllReduce {
public:
  Ring_AllReduce(Transport t, Device d, int r, int N)
      : t_(t), d_(d), rank_(r), world_size_(N) {}
  void execute(ValueType *data, std::size_t total_elements);

private:
  Transport t_;
  Device d_;
  int rank_;
  int world_size_;
};

#endif