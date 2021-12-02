// clang-format off
// RUN: mlir-clang %std %s | runner -ptr2tsr | FileCheck %s

#include "dsl.h"

int main() {
  float din[2 * 3 * 4 * 5], dout[2 * 6 * 4 * 5];
  Tensor A = tensor_load(din);
  A = tensor_reshape4d(A, 2, 3, 4, 5);
  A = tensor_concat(1, A, A);
  A = tensor_reshape1d(A, 2 * 6 * 4 * 5);
  tensor_store(A, dout);

  return 0;
}

// CHECK: %[[v4:.*]] = "tosa.reshape"(%{{.*}}) {new_shape = [2, 3, 4, 5]} : (tensor<?xf32>) -> tensor<2x3x4x5xf32>
// CHECK: %[[v5:.*]] = "tosa.concat"(%[[v4]], %[[v4]]) {axis = 1 : i64} : (tensor<2x3x4x5xf32>, tensor<2x3x4x5xf32>) -> tensor<?x?x?x?xf32>
// CHECK: %[[v6:.*]] = "tosa.reshape"(%[[v5]]) {new_shape = [240]} : (tensor<?x?x?x?xf32>) -> tensor<240xf32>
