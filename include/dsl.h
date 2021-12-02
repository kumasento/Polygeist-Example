#ifndef DSL_H
#define DSL_H

#include <cstdint>

#define Tensor void *

template <typename T> Tensor tensor_load(T);        // -> memref.tensor_load
template <typename T> void tensor_store(Tensor, T); // -> memref.tensor_store

// -> tosa.concat
template <typename... Args> Tensor tensor_concat(int64_t, Args... args);

extern "C" {
Tensor tensor_add(Tensor, Tensor); // -> tosa.add
// (input, weight, bias, pad0, pad1, pad2, pad3, stride0, stride1, dilation0,
// dilation1)
Tensor tensor_conv2d(Tensor, Tensor, Tensor, int64_t, int64_t, int64_t, int64_t,
                     int64_t, int64_t, int64_t, int64_t); // -> tosa.conv2d
Tensor tensor_reshape1d(Tensor, int64_t);
Tensor tensor_reshape4d(Tensor, int64_t, int64_t, int64_t, int64_t);
}

#endif
