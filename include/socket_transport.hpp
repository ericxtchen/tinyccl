// This file is for socket transfer only without a CUDA GPU
#include <cstddef>

class SocketTransfer {
private:
  const int BASE_PORT;
  const int rank_;
  const int world_size_; // the total number of processes
  int sock_send_ = -1;
  int sock_recv_ = -1;

public:
  SocketTransfer(const int BASE_PORT, const int rank, const int N);
  ~SocketTransfer();
  void send(const void *send_data, std::size_t bytes, int dest_rank);
  void recv(void *recv_data, std::size_t bytes, int src_rank);
  void sendrecv(const void *send_data, std::size_t dest_bytes, int dest_rank,
                void *recv_data, std::size_t src_bytes, int src_rank);
};