// RUN: mlir-clang %std %s | runner -map-to-dialects | FileCheck %s
#define Tensor void *

template <typename T> Tensor tensor_load(T);        // -> memref.tensor_load
template <typename T> void tensor_store(Tensor, T); // -> memref.tensor_store

int main() {
  float din[128], dout[128];
  Tensor A = tensor_load(din);
  tensor_store(A, dout);

  return 0;
}

// CHECK: func @main()
// CHECK:  %[[v0:.*]] = memref.alloca() : memref<128xf32>
// CHECK:  %[[v1:.*]] = memref.alloca() : memref<128xf32>
// CHECK:  %[[v2:.*]] = memref.cast %[[v1]] : memref<128xf32> to memref<?xf32>
// CHECK:  %[[v3:.*]] = memref.tensor_load %[[v2]] : memref<?xf32>
// CHECK:  %[[v4:.*]] = memref.cast %[[v0]] : memref<128xf32> to memref<?xf32>
// CHECK:  memref.tensor_store %[[v3]], %[[v4]] : memref<?xf32>
