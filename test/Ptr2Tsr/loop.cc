// RUN: mlir-clang %std %s | runner -ptr2tsr | FileCheck %s

#include "dsl.h"

int main() {
  float din[128], dout[128];
  Tensor A = tensor_load(din);
  for (int i = 0; i < 2; ++i) {
    A = tensor_add(A, A);
    for (int j = 0; j < 4; ++j)
      A = tensor_add(A, A);
  }
  tensor_store(A, dout);
  return 0;
}

// clang-format off
// CHECK: %[[v0:.*]] = memref.alloca() : memref<128xf32>
// CHECK: %[[v1:.*]] = memref.alloca() : memref<128xf32>
// CHECK: %[[v2:.*]] = memref.cast %[[v1]] : memref<128xf32> to memref<?xf32>
// CHECK: %[[v3:.*]] = memref.tensor_load %[[v2]] : memref<?xf32>
// CHECK: %[[v4:.*]] = scf.for %{{.*}} iter_args(%[[arg1:.*]] = %[[v3]]) -> (tensor<?xf32>) 
// CHECK:   %[[v6:.*]] = "tosa.add"(%[[arg1]], %[[arg1]]) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK:   %[[v7:.*]] = scf.for %{{.*}} iter_args(%[[arg3:.*]] = %[[v6]]) -> (tensor<?xf32>) {
// CHECK:     %[[v8:.*]] = "tosa.add"(%[[arg3]], %[[arg3]]) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK:     scf.yield %[[v8]] : tensor<?xf32>
// CHECK:   }
// CHECK:   scf.yield %[[v7]] : tensor<?xf32>
// CHECK: }
// CHECK: %[[v5:.*]] = memref.cast %[[v0]] : memref<128xf32> to memref<?xf32>
// CHECK: memref.tensor_store %[[v4]], %[[v5]] : memref<?xf32>
// clang-format on
