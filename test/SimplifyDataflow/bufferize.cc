// RUN: mlir-clang %std %s | runner -simplify-dataflow | FileCheck %s
#include "dsl.h"

int main() {
  float din[128], dout[128];
  tensor_store(tensor_add(tensor_load(din), tensor_load(din)), dout);

  return 0;
}

// CHECK: func @main()
// CHECK: %{{.*}} = llvm.alloca %c1_i64 x !llvm.ptr<i8>
// CHECK: %{{.*}} = llvm.alloca %c1_i64 x !llvm.ptr<i8>
// CHECK: %{{.*}} = llvm.alloca %c1_i64 x !llvm.ptr<i8>
