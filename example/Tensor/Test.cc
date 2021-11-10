#define Tensor void*

template <typename T>
Tensor tensor_load(T);  // -> memref.tensor_load
template <typename T>
void tensor_store(Tensor, T);  // -> memref.tensor_store

extern "C" {
Tensor tensor_add(Tensor, Tensor);  // -> tosa.add
void prevent_dce(Tensor);           // -> nothing
}

int main() {
  float din[128], dout[128];
  Tensor A = tensor_load(din);
  Tensor B = tensor_add(A, A);
  tensor_store(B, dout);

  return 0;
}
