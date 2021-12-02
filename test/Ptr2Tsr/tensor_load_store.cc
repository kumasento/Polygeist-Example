// RUN: mlir-clang %std %s | runner -ptr2tsr  | FileCheck %s

#include "dsl.h"

int main() {
  float din[128], dout[128];
  Tensor A = tensor_load(din);
  tensor_store(A, dout);

  return 0;
}

// CHECK: func @main()
// CHECK:   %[[c0_i32:.*]] = arith.constant 0 : i32
// CHECK:   %[[c1_i64:.*]] = arith.constant 1 : i64
// CHECK:   %[[v0:.*]] = memref.alloca() : memref<128xf32>
// CHECK:   %[[v1:.*]] = memref.alloca() : memref<128xf32>
// CHECK:   %[[v2:.*]] = memref.cast %[[v1]] : memref<128xf32> to memref<?xf32>
// CHECK:   %[[v3:.*]] = memref.tensor_load %[[v2]] : memref<?xf32>
// CHECK:   %[[v4:.*]] = memref.cast %[[v0]] : memref<128xf32> to memref<?xf32>
// CHECK:   memref.tensor_store %[[v3]], %[[v4]] : memref<?xf32>
