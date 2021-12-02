// RUN: mlir-clang %std %s | runner -ptr2tsr | FileCheck %s

#include "dsl.h"

int main() {
  float din[128], dout[128];
  Tensor A = tensor_load(din);
  Tensor B = tensor_reshape1d(A, 128);
  tensor_store(B, dout);
  return 0;
}

// CHECK: %[[v0:.*]] = memref.alloca() : memref<128xf32>
// CHECK: %[[v1:.*]] = memref.alloca() : memref<128xf32>
// CHECK: %[[v2:.*]] = memref.cast %[[v1]] : memref<128xf32> to memref<?xf32>
// CHECK: %[[v3:.*]] = memref.tensor_load %[[v2]] : memref<?xf32>
// CHECK: %[[v4:.*]] = "tosa.reshape"(%[[v3]]) {new_shape = [128]}
// CHECK: %[[v5:.*]] = memref.cast %[[v0]] : memref<128xf32> to memref<?xf32>
// CHECK: %[[v6:.*]] = tensor.cast %[[v4]] : tensor<128xf32> to tensor<?xf32>
// CHECK: memref.tensor_store %[[v6]], %[[v5]] : memref<?xf32>
