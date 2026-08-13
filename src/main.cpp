#include "../include/cpu_device.hpp"
#include "../include/ring_allreduce.hpp"
#include "../include/socket_transport.hpp"
#include <array>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "You must provide the rank and the world size.\n";
    return 1;
  }
  std::array<float, 5> arr = {1.0, 2.0, 3.0, 4.0, 5.0};
  constexpr int BASE_PORT = 8000;
  int rank = std::stoi(argv[1]);
  int world_size = std::stoi(argv[2]);

  SocketTransport t(BASE_PORT, rank, world_size);
  CpuDevice<float> d{};
  Ring_AllReduce<float, SocketTransport, CpuDevice<float>> allreduce(
      t, d, rank, world_size);
  allreduce.execute(arr.data(), arr.size());

  std::cout << "Rank " << rank << " result: ";
  for (float v : arr)
    std::cout << v << " ";
  std::cout << "\n";

  return 0;
}