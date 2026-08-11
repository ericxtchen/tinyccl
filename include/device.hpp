// this represents the device concept (CPU or GPU/CUDA)
#ifndef DEVICE_H
#define DEVICE_H

#include <concepts>

template <typename D, typename ValueType>
concept DevicePolicy = requires(D &d, const ValueType *src_buffer,
                                ValueType *dest_buffer, std::size_t count) {
  { d.reduce_add(src_buffer, dest_buffer, count) } -> std::same_as<void>;
};

#endif