#include "../include/device.hpp"
#include "../include/transport.hpp"

template <typename ValueType, Transport transport, Device<ValueType> device>
class Ring_AllReduce {
public:
  Ring_AllReduce(transport t, device d, int r, int N);
  void execute();

private:
  transport t_;
  device d_;
  int rank_;
  int world_size_;
};