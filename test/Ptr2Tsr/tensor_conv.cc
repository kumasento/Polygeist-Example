// RUN: mlir-clang %std %s | runner -ptr2tsr | FileCheck %s
#include "dsl.h"

int main() {
  float input[2 * 3 * 32 * 32], weight[4 * 3 * 3 * 3], bias[4],
      output[2 * 4 * 32 * 32];
  Tensor X = tensor_load(input);
  Tensor W = tensor_load(weight);
  Tensor b = tensor_load(bias);

  X = tensor_reshape4d(X, 2, 3, 32, 32);
  W = tensor_reshape4d(W, 4, 3, 3, 3);
  b = tensor_reshape1d(b, 4);

  Tensor Y = tensor_conv2d(X, W, b, 1, 1, 1, 1, 1, 1, 1, 1);
  Y = tensor_reshape1d(Y, -1);
  tensor_store(Y, output);
  return 0;
}

// CHECK: func @main
// CHECK: %[[v0:.*]] = memref.alloca() : memref<8192xf32>
// CHECK: %[[v1:.*]] = memref.alloca() : memref<4xf32>
// CHECK: %[[v2:.*]] = memref.alloca() : memref<108xf32>
// CHECK: %[[v3:.*]] = memref.alloca() : memref<6144xf32>
// CHECK: %[[v4:.*]] = memref.cast %[[v3]] : memref<6144xf32> to memref<?xf32>
// CHECK: %[[v5:.*]] = memref.tensor_load %[[v4]] : memref<?xf32>
// CHECK: %[[v6:.*]] = memref.cast %[[v2]] : memref<108xf32> to memref<?xf32>
// CHECK: %[[v7:.*]] = memref.tensor_load %[[v6]] : memref<?xf32>
// CHECK: %[[v8:.*]] = memref.cast %[[v1]] : memref<4xf32> to memref<?xf32>
// CHECK: %[[v9:.*]] = memref.tensor_load %[[v8]] : memref<?xf32>
// CHECK: %[[v10:.*]] = "tosa.reshape"(%[[v5]]) {new_shape = [2, 3, 32, 32]}
// CHECK: %[[v11:.*]] = "tosa.reshape"(%[[v7]]) {new_shape = [4, 3, 3, 3]}
// CHECK: %[[v12:.*]] = "tosa.reshape"(%[[v9]]) {new_shape = [4]}
// CHECK: %[[v13:.*]] = "tosa.conv2d"(%[[v10]], %[[v11]], %[[v12]])
// CHECK: {dilation = [1, 1], pad = [1, 1, 1, 1], stride = [1, 1]}
// CHECK: %[[v14:.*]] = "tosa.reshape"(%[[v13]]) {new_shape = [-1]}
// CHECK: %[[v15:.*]] = memref.cast %[[v0]] : memref<8192xf32> to memref<?xf32>
// CHECK: memref.tensor_store %[[v14]], %[[v15]] : memref<?xf32>
