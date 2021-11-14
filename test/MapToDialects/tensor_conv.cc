// RUN: exit 0
#include <cstdint>

#define Tensor void *

template <typename T> Tensor tensor_load(T);        // -> memref.tensor_load
template <typename T> void tensor_store(Tensor, T); // -> memref.tensor_store

extern "C" {
// (input, weight, bias, pad0, pad1, pad2, pad3, stride0, stride1, dilation0,
// dilation1)
Tensor tensor_conv2d(Tensor, Tensor, Tensor, int64_t, int64_t, int64_t, int64_t,
                     int64_t, int64_t, int64_t, int64_t); // -> tosa.conv2d
Tensor tensor_set_shape1d(Tensor, int64_t);
Tensor tensor_set_shape4d(Tensor, int64_t, int64_t, int64_t, int64_t);
}

int main() {
  float input[2 * 3 * 32 * 32], weight[4 * 3 * 3 * 3], bias[4],
      output[2 * 4 * 32 * 32];
  Tensor X = tensor_load(input);
  Tensor W = tensor_load(weight);
  Tensor b = tensor_load(bias);

  X = tensor_set_shape4d(X, 2, 3, 32, 32);
  W = tensor_set_shape4d(W, 4, 3, 3, 3);
  b = tensor_set_shape1d(b, 4);

  Tensor Y = tensor_conv2d(tensor_set_shape4d(X, 2, 3, 32, 32), W, b, 1, 1, 1,
                           1, 1, 1, 1, 1);
  tensor_store(Y, output);
}
