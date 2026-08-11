#include "../include/cpu_device.hpp"
template <typename T>
void CpuDevice<T>::reduce_add(const T *src_buffer, T *dest_buffer,
                              std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    dest_buffer[i] += src_buffer[i];
  }
};