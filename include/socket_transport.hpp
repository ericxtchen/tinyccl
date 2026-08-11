// This file is for socket transfer only without a CUDA GPU
#ifndef SOCKET_TRANSPORT_H
#define SOCKET_TRANSPORT_H

#include <cstddef>

class SocketTransport {
private:
  const int BASE_PORT;
  const int rank_;
  const int world_size_; // the total number of processes
  int sock_send_ = -1;
  int sock_recv_ = -1;

public:
  SocketTransport(const int BASE_PORT, const int rank, const int N);
  ~SocketTransport();
  void send(const void *send_data, std::size_t bytes, int dest_rank);
  void recv(void *recv_data, std::size_t bytes, int src_rank);
  void sendrecv(const void *send_data, std::size_t dest_bytes, int dest_rank,
                void *recv_data, std::size_t src_bytes, int src_rank);
};

#endif