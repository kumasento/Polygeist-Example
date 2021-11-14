// RUN: mlir-clang %std %s | runner -simplify-dataflow | FileCheck %s
#define Tensor void *

template <typename T> Tensor tensor_load(T);        // -> memref.tensor_load
template <typename T> void tensor_store(Tensor, T); // -> memref.tensor_store

extern "C" {
Tensor tensor_add(Tensor, Tensor); // -> tosa.add
}

int main() {
  float din[128], dout[128];
  tensor_store(tensor_add(tensor_load(din), tensor_load(din)), dout);

  return 0;
}

// CHECK: func @main()
// CHECK: %{{.*}} = llvm.alloca %c1_i64 x !llvm.ptr<i8>
// CHECK: %{{.*}} = llvm.alloca %c1_i64 x !llvm.ptr<i8>
// CHECK: %{{.*}} = llvm.alloca %c1_i64 x !llvm.ptr<i8>
