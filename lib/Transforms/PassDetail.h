//===- PassDetail.h - Transforms Pass class details -------------*- C++ -*-===//

#ifndef PGEX_MLIR_TRANSFORMS_PASSDETAIL_H_
#define PGEX_MLIR_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"
#include "pgex/Transforms/Passes.h"

namespace pgex {
#define GEN_PASS_CLASSES
#include "pgex/Transforms/Passes.h.inc"
}  // namespace pgex

#endif  // TRANSFORMS_PASSDETAIL_H_
