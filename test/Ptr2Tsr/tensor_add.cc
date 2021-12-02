// RUN: mlir-clang %std %s | runner -ptr2tsr | FileCheck %s
#include "dsl.h"

int main() {
  float din[128], dout[128];
  Tensor A = tensor_load(din);
  Tensor B = tensor_add(A, A);
  tensor_store(B, dout);

  return 0;
}

// CHECK: func @main()
// CHECK:  %[[v0:.*]] = memref.alloca() : memref<128xf32>
// CHECK:  %[[v1:.*]] = memref.alloca() : memref<128xf32>
// CHECK:  %[[v2:.*]] = memref.cast %[[v1]] : memref<128xf32> to memref<?xf32>
// CHECK:  %[[v3:.*]] = memref.tensor_load %[[v2]] : memref<?xf32>
// CHECK:  %[[v4:.*]] = "tosa.add"(%[[v3]], %[[v3]])
// CHECK:  %[[v5:.*]] = memref.cast %[[v0]] : memref<128xf32> to memref<?xf32>
// CHECK:  memref.tensor_store %[[v4]], %[[v5]] : memref<?xf32>
