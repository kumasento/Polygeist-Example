# Polygeist-Example

Demonstrating what you can do with Polygeist.

## Compile a simple C function

Input:

```c
/// SimpleCFunc.c
void MatrixMult(float A[100][200], float B[200][300], float C[100][300]) {
  int i, j, k;

  for (i = 0; i < 100; i ++) {
    for (j = 0; j < 300; j ++) {
      C[i][j] = 0;
      for (k = 0; k < 200; k ++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
```

Run:

```
mlir-clang --function=MatrixMult --raise-scf-to-affine SimpleCFunc.c
```

You get:

```mlir
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @MatrixMult(%arg0: memref<?x200xf32>, %arg1: memref<?x300xf32>, %arg2: memref<?x300xf32>) {
    %c0_i32 = constant 0 : i32
    %0 = sitofp %c0_i32 : i32 to f32
    affine.for %arg3 = 0 to 100 {
      affine.for %arg4 = 0 to 300 {
        affine.store %0, %arg2[%arg3, %arg4] : memref<?x300xf32>
        affine.for %arg5 = 0 to 200 {
          %1 = affine.load %arg0[%arg3, %arg5] : memref<?x200xf32>
          %2 = affine.load %arg1[%arg5, %arg4] : memref<?x300xf32>
          %3 = mulf %1, %2 : f32
          %4 = affine.load %arg2[%arg3, %arg4] : memref<?x300xf32>
          %5 = addf %4, %3 : f32
          affine.store %5, %arg2[%arg3, %arg4] : memref<?x300xf32>
        }
      }
    }
    return
  }
}
```

## External C function calls

Input: 

```c
#include <stdlib.h>

struct TensorRecord { float *data; size_t size; };

typedef struct TensorRecord *Tensor;

extern Tensor Add(Tensor x, Tensor y);
extern Tensor Mul(Tensor x, Tensor y);

Tensor Compute(Tensor x, Tensor y) {
  return Add(Mul(x, x), Mul(y, y));
}
```

```mlir
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @Compute(%arg0: memref<?x!llvm.struct<(memref<?xf32>, i64)>>, %arg1: memref<?x!llvm.struct<(memref<?xf32>, i64)>>) -> memref<?x!llvm.struct<(memref<?xf32>, i64)>> {
    %0 = call @Mul(%arg0, %arg0) : (memref<?x!llvm.struct<(memref<?xf32>, i64)>>, memref<?x!llvm.struct<(memref<?xf32>, i64)>>) -> memref<?x!llvm.struct<(memref<?xf32>, i64)>>
    %1 = call @Mul(%arg1, %arg1) : (memref<?x!llvm.struct<(memref<?xf32>, i64)>>, memref<?x!llvm.struct<(memref<?xf32>, i64)>>) -> memref<?x!llvm.struct<(memref<?xf32>, i64)>>
    %2 = call @Add(%0, %1) : (memref<?x!llvm.struct<(memref<?xf32>, i64)>>, memref<?x!llvm.struct<(memref<?xf32>, i64)>>) -> memref<?x!llvm.struct<(memref<?xf32>, i64)>>
    return %2 : memref<?x!llvm.struct<(memref<?xf32>, i64)>>
  }
  func private @Add(memref<?x!llvm.struct<(memref<?xf32>, i64)>>, memref<?x!llvm.struct<(memref<?xf32>, i64)>>) -> memref<?x!llvm.struct<(memref<?xf32>, i64)>>
  func private @Mul(memref<?x!llvm.struct<(memref<?xf32>, i64)>>, memref<?x!llvm.struct<(memref<?xf32>, i64)>>) -> memref<?x!llvm.struct<(memref<?xf32>, i64)>>
}
```

## Control Flow

Input:

```c
#include <stdlib.h>

struct TensorRecord { float *data; size_t size; };

typedef struct TensorRecord *Tensor;

extern Tensor Add(Tensor, Tensor);

Tensor Reduction(Tensor init, Tensor x, int N) {
  int i;
  Tensor y = init;
  for (i = 0; i < N; i ++)
    y = Add(y, x);
  return y;
}
```

Output:

```mlir
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @Reduction(%arg0: memref<?x!llvm.struct<(memref<?xf32>, i64)>>, %arg1: memref<?x!llvm.struct<(memref<?xf32>, i64)>>, %arg2: i32) -> memref<?x!llvm.struct<(memref<?xf32>, i64)>> {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = index_cast %arg2 : i32 to index
    %1 = scf.for %arg3 = %c0 to %0 step %c1 iter_args(%arg4 = %arg0) -> (memref<?x!llvm.struct<(memref<?xf32>, i64)>>) {
      %2 = call @Add(%arg4, %arg1) : (memref<?x!llvm.struct<(memref<?xf32>, i64)>>, memref<?x!llvm.struct<(memref<?xf32>, i64)>>) -> memref<?x!llvm.struct<(memref<?xf32>, i64)>>
      scf.yield %2 : memref<?x!llvm.struct<(memref<?xf32>, i64)>>
    }
    return %1 : memref<?x!llvm.struct<(memref<?xf32>, i64)>>
  }
  func private @Add(memref<?x!llvm.struct<(memref<?xf32>, i64)>>, memref<?x!llvm.struct<(memref<?xf32>, i64)>>) -> memref<?x!llvm.struct<(memref<?xf32>, i64)>>
}
```
