#ifndef CPUDEVICE_H
#define CPUDEVICE_H

#include <cstddef>

template <typename T> class CpuDevice {
public:
  void reduce_add(const T *src_buffer, T *dest_buffer, std::size_t count);
};

#endif