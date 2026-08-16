#ifndef TRANSPORT_H
#define TRANSPORT_H

#include <concepts>

template <typename T>
concept TransportPolicy = requires(T &t, const void *send_data, void *recv_data,
                                   std::size_t bytes, int rank) {
  { t.send(send_data, bytes, rank) } -> std::same_as<void>;
  { t.recv(recv_data, bytes, rank) } -> std::same_as<void>;
  {
    t.sendrecv(send_data, bytes, rank, recv_data, bytes, rank)
  } -> std::same_as<void>;
};

#endif