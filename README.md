# Polygeist Example

[[Examples](example)]

## Download

```sh
git clone --recursive https://github.com/kumasento/Polygeist-Example
cd Polygeist-Example
```

## Build Polygeist

Checkout [here](https://github.com/wsmoses/Polygeist).


## Build this project

Make sure you have Ninja, CMake, and the Clang toolchain installed.

```sh
mkdir build
cd build

cmake .. -G Ninja \
  -DLLVM_USE_LINKER=lld \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DMLIR_DIR=${PWD}/../polygeist/llvm-project/build/lib/cmake/mlir \
  -DCMAKE_BUILD_TYPE=DEBUG \
  -DLLVM_ENABLE_ASSERTIONS=ON

ninja
```


## Example: mapping to TOSA and memref dialect

```c++
#define Tensor void *

template <typename T> Tensor tensor_load(T);        // -> memref.tensor_load
template <typename T> void tensor_store(Tensor, T); // -> memref.tensor_store

extern "C" {
Tensor tensor_add(Tensor, Tensor); // -> tosa.add
}

int main() {
  float din[128], dout[128];
  Tensor A = tensor_load(din);
  Tensor B = tensor_add(A, A);
  tensor_store(B, dout);

  return 0;
}
```

```sh
# Under /build
../polygeist/build/mlir-clang/mlir-clang -S -O0 ../test/MapToDialects/tensor_add.cc | ./bin/runner --map-to-dialects
```

```mlir
func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %0 = memref.alloca() : memref<128xf32>
  %1 = memref.alloca() : memref<128xf32>
  %2 = memref.cast %1 : memref<128xf32> to memref<?xf32>
  %3 = memref.tensor_load %2 : memref<?xf32>
  %4 = "tosa.add"(%3, %3) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %5 = memref.cast %0 : memref<128xf32> to memref<?xf32>
  memref.tensor_store %4, %5 : memref<?xf32>
  return %c0_i32 : i32
}
```

The key pass is `-map-to-dialects` as implemented [here](./lib/Transforms/MapToDialects.cc).
