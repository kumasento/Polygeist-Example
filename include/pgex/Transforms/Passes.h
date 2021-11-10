#ifndef PGEX_MLIR_TRANSFORMS_PASSES_H
#define PGEX_MLIR_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace pgex {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createMapToDialectsPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "pgex/Transforms/Passes.h.inc"

}  // namespace pgex

#endif
