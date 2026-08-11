#ifndef TRANSPORT_H
#define TRANSPORT_H

#include <cstddef>

template <typename T>
concept Transport = requires(T &t, const void *send_data, void *recv_data,
                             std::size_t bytes, int rank) {
  t.send(send_data, bytes, rank);
  t.recv(recv_data, bytes, rank);
};

#endif