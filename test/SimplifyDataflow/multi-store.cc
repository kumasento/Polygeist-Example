// RUN: mlir-clang %std %s | runner -simplify-dataflow='duplicate=1' | FileCheck %s

#include "dsl.h"

int main() {
  float din[128], dout[128];
  Tensor A = tensor_load(din);
  A = tensor_add(A, A);
  tensor_store(A, dout);

  return 0;
}

// CHECK: func @main()
// CHECK: %{{.*}} = llvm.alloca %c1_i64 x !llvm.ptr<i8>
// CHECK: %{{.*}} = llvm.alloca %c1_i64 x !llvm.ptr<i8>
