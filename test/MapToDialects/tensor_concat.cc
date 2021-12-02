// RUN: exit 0
#include <cstdint>

#define Tensor void *

template <typename T> Tensor tensor_load(T);        // -> memref.tensor_load
template <typename T> void tensor_store(Tensor, T); // -> memref.tensor_store

// -> tosa.concat
template <typename... Ts> Tensor tensor_concat(int64_t, Ts...);

extern "C" {
Tensor tensor_reshape4d(Tensor, int64_t, int64_t, int64_t, int64_t);
}

int main() {

  float din[2 * 3 * 4 * 4], dout[2 * 12 * 4 * 4];
  Tensor A = tensor_reshape4d(tensor_load(din), 2, 3, 4, 4);
  for (int i = 0; i < 2; i++)
    A = tensor_concat(1, A, A);

  tensor_store(A, dout);
  return 0;
}
