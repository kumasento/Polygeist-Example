// RUN: exit 0

#include "dsl.h"

int main() {
  float din[128], dout[128];
  Tensor A = tensor_load(din);
  for (int i = 0; i < 2; ++i)
    A = tensor_add(A, A);
  tensor_store(A, dout);
  return 0;
}

// CHECK: %[[v8:.*]] = llvm.load %[[v0:.*]] : !llvm.ptr<ptr<i8>>
// CHECK: llvm.store %[[v8]], %[[v2:.*]] : !llvm.ptr<ptr<i8>>
// CHECK: scf.for
// CHECK:   %[[v12:.*]] = llvm.load %[[v2]] : !llvm.ptr<ptr<i8>>
// CHECK:   %[[v13:.*]] = llvm.load %[[v2]] : !llvm.ptr<ptr<i8>>
// CHECK:   %[[v14:.*]] = call @tensor_add(%[[v12]], %[[v13]]) : (!llvm.ptr<i8>,
// !llvm.ptr<i8>) -> !llvm.ptr<i8> CHECK:   llvm.store %[[v14]], %[[v3]] :
// !llvm.ptr<ptr<i8>> CHECK: %[[v9:.*]] = llvm.load %[[v2]] : !llvm.ptr<ptr<i8>>
// CHECK: llvm.store %[[v9]], %[[v1 : !llvm.ptr<ptr<i8>>
