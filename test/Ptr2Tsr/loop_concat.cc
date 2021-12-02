// RUN: mlir-clang %std %s | runner -ptr2tsr | FileCheck %s

#include "dsl.h"

int main() {
  float din[1 * 2 * 3 * 4], dout[1 * 12 * 3 * 4];
  Tensor A = tensor_load(din);
  A = tensor_reshape4d(A, 1, 2, 3, 4);
  for (int i = 0; i < 2; ++i)
    A = tensor_concat(2, A, A);
  A = tensor_reshape1d(A, 1 * 12 * 3 * 4);
  tensor_store(A, dout);
  return 0;
}

// clang-format off
// CHECK: %[[v0:.*]] = memref.alloca() : memref<144xf32>
// CHECK: %[[v1:.*]] = memref.alloca() : memref<24xf32>
// CHECK: %[[v2:.*]] = memref.cast %[[v1]] : memref<24xf32> to memref<?xf32>
// CHECK: %[[v3:.*]] = memref.tensor_load %[[v2]] : memref<?xf32>
// CHECK: %[[v4:.*]] = "tosa.reshape"(%[[v3]]) {new_shape = [1, 2, 3, 4]} : (tensor<?xf32>) -> tensor<1x2x3x4xf32>
// CHECK: %[[v5:.*]] = tensor.cast %[[v4]] : tensor<1x2x3x4xf32> to tensor<?x?x?x?xf32>
// CHECK: %[[v6:.*]] = scf.for %{{.*}} iter_args(%[[arg1:.*]] = %[[v5]]) -> (tensor<?x?x?x?xf32>) 
// CHECK:   %[[v10:.*]] = "tosa.concat"(%[[arg1]], %[[arg1]]) {axis = 2 : i64} : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:   scf.yield %[[v10]] : tensor<?x?x?x?xf32>
// CHECK: %[[v7:.*]] = "tosa.reshape"(%[[v6]]) {new_shape = [144]} : (tensor<?x?x?x?xf32>) -> tensor<144xf32>
// CHECK: %[[v8:.*]] = memref.cast %[[v0]] : memref<144xf32> to memref<?xf32>
// CHECK: %[[v9:.*]] = tensor.cast %[[v7]] : tensor<144xf32> to tensor<?xf32>
// CHECK: memref.tensor_store %[[v9]], %[[v8]] : memref<?xf32>
