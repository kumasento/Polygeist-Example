#!/usr/bin/env bash

set -x
set -e
set -o pipefail

LLVMDIR="$1"
export PATH="${LLVMDIR}/bin:${PATH}"

SRCMAIN=TestMain.c
SRCLIB=Test.c
SRCMLIR=Lib.mlir

# Compile MLIR library to LLVM IR
mlir-opt -convert-std-to-llvm "${SRCMLIR}" | mlir-translate -mlir-to-llvmir > "${SRCMLIR%.mlir}.ll"

# Compile C to MLIR
mlir-clang --function=TestAdd "${SRCLIB}" -I="${LLVMDIR}/lib/clang/13.0.0/include" > "${SRCLIB%.c}.mlir"
# Then to LLVM.
mlir-opt -convert-std-to-llvm "${SRCLIB%.c}.mlir" | mlir-translate -mlir-to-llvmir > "${SRCLIB%.c}.ll"

# Compile LLVM IR to a shared library
clang "${SRCMLIR%.mlir}.ll" "${SRCLIB%.c}.ll" -shared -o "${SRCLIB%.c}.so"

# Finally compile with the main file
clang "${SRCMAIN}" "${SRCLIB%.c}.so" -o "${SRCMAIN%.c}"

./"${SRCMAIN%.c}"
