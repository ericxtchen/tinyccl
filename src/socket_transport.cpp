#include "../include/socket_transport.hpp"
#include <arpa/inet.h>
#include <cassert>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

// ideally, we should have a connection to our left neighbor to receive data and
// our right neighbor to send data without causing any deadlocks or race
// conditions with any blocking socket operations... we should also pair a send
// and recv operation to avoid deadlock issues if we follow the SPMD model, we
// should be pairing it each, but what happens when we need to reduce? what
// should be the order of operations? if we follow SPMD model, that mean each
// process is doing the same thing simultaneously right? does that mean we have
// to wait for the slowest process to finish some step like recv, send, or
// reduce? how will that work?

/**
        We keep looping to connect to our send neighbor, while trying to wait
   for a recv connection from our recv neighbor With the SPMD model, this is
   done by every process, so every node in the ring tries to actively connect to
   the right neighbor and listen to the left neighbor and it all comes back
   together because of the ring structure.
*/
SocketTransfer::SocketTransfer(int rank, int N, int base_port)
    : rank_(rank), world_size_(N), BASE_PORT(base_port) {

  int my_listen_port = BASE_PORT + rank;
  int next_rank_port = BASE_PORT + ((rank + 1) % N);

  // Create the listening socket
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  sockaddr_in server_addr{};
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = INADDR_ANY;
  server_addr.sin_port = htons(my_listen_port);

  if (bind(listen_fd, (struct sockaddr *)&server_addr, sizeof(server_addr)) <
      0) {
    throw std::runtime_error("Bind failed on port " +
                             std::to_string(my_listen_port));
  }
  listen(listen_fd, 1);

  // Spawn a thread to accept the incoming connection from prev_rank
  // (We do this in a thread so we don't block our own outbound connection
  // attempt)
  std::thread accept_thread([this, listen_fd]() {
    sockaddr_in client_addr{};
    socklen_t client_len = sizeof(client_addr);

    // This blocks until prev_rank successfully connects to us
    this->sock_recv_ =
        accept(listen_fd, (struct sockaddr *)&client_addr, &client_len);

    // We have the connection, we don't need to listen for anyone else
    close(listen_fd);
  });

  // Loop attempting to connect to next_rank
  this->sock_send_ = socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in dest_addr{};
  dest_addr.sin_family = AF_INET;
  inet_pton(AF_INET, "127.0.0.1",
            &dest_addr.sin_addr); // Assuming localhost for now
  dest_addr.sin_port = htons(next_rank_port);

  while (true) {
    if (connect(this->sock_send_, (struct sockaddr *)&dest_addr,
                sizeof(dest_addr)) == 0) {
      break; // Connection succeeded!
    }
    // Connection failed (next_rank isn't listening yet). Sleep and retry.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // Wait for our accept thread to finish getting sock_prev
  accept_thread.join();

  std::cout << "Rank " << rank << " network initialized. "
            << "sock_next -> Rank " << (rank + 1) % N << " | sock_prev <- Rank "
            << (rank - 1 + N) % N << "\n";
}

SocketTransfer::~SocketTransfer() {
  if (sock_send_ != -1) {
    close(sock_send_);
  }

  if (sock_recv_ != -1) {
    close(sock_recv_);
  }
}

void SocketTransfer::sendrecv(const void *send_data, std::size_t send_bytes,
                              int dest_rank, void *recv_data,
                              std::size_t recv_bytes, int src_rank) {
  std::thread send_thread([this, send_data, send_bytes, dest_rank]() {
    send(send_data, send_bytes, dest_rank);
  });

  this->recv(recv_data, recv_bytes, src_rank);
  send_thread.join();
}

void send_all(int sock_fd, const void *send_data, std::size_t bytes) {
  const char *ptr = static_cast<const char *>(send_data);
  std::size_t total_sent = 0;
  while (total_sent < bytes) {
    ssize_t sent = ::send(sock_fd, ptr + total_sent, bytes - total_sent, 0);
    if (sent <= 0)
      break;
    total_sent += sent;
  }
}

void SocketTransfer::send(const void *send_data, std::size_t bytes, int rank) {
  // verify that rank maps to sock_next
  assert(rank == (this->rank_ + 1) % this->world_size_ &&
         "dest_rank does not match sock_next rank!");
  send_all(this->sock_send_, send_data, bytes);
}

std::vector<float> recv_all(int recv_fd, void *recv_data, std::size_t bytes) {
  if (bytes % sizeof(float) != 0)
    throw std::runtime_error("byte count is not a multiple of sizeof(float)");

  std::vector<float> floats(bytes / sizeof(float));
  std::byte *data = reinterpret_cast<std::byte *>(floats.data());

  std::size_t received = 0;

  while (received < bytes) {
    ssize_t n = recv(recv_fd, data + received, bytes - received, 0);

    if (n == 0) {
      throw std::runtime_error("peer closed connection");
    }

    if (n < 0) {
      throw std::runtime_error("recv() failed");
    }

    received += static_cast<std::size_t>(n);
  }

  return floats;
}

void SocketTransfer::recv(void *recv_data, std::size_t bytes, int src_rank) {
  assert(src_rank ==
             (this->rank_ - 1 + this->world_size_) % this->world_size_ &&
         "src_rank does not match sock_prev rank!");
  recv_all(this->sock_recv_, recv_data, bytes);
}