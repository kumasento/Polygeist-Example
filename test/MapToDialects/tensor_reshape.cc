// RUN: mlir-clang %std %s | runner -map-to-dialects | FileCheck %s

#include "dsl.h"

int main() {
  float din[128], dout[128];
  Tensor A = tensor_load(din);
  Tensor B = tensor_reshape1d(A, 128);
  tensor_store(B, dout);
  return 0;
}

// CHECK:  func @main()
// CHECK-DAG:    %[[dout:.*]] = memref.alloca() : memref<128xf32>
// CHECK-DAG:    %[[din:.*]] = memref.alloca() : memref<128xf32>
// CHECK:    %[[v2:.*]] = memref.cast %[[din]] : memref<128xf32> to memref<?xf32>
// CHECK:    %[[v3:.*]] = memref.tensor_load %[[v2]] : memref<?xf32>
// CHECK:    %[[v4:.*]] = "tosa.reshape"(%[[v3]]) {new_shape = [128]} : (tensor<?xf32>) -> tensor<128xf32>
// CHECK:    %[[v5:.*]] = memref.cast %[[dout]] : memref<128xf32> to memref<?xf32>
// CHECK:    %[[v6:.*]] = memref.cast %[[v5]] : memref<?xf32> to memref<128xf32>
// CHECK:    memref.tensor_store %[[v4]], %[[v6]] : memref<128xf32>
