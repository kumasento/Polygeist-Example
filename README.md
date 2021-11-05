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

