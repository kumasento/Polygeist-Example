#ifndef PGEX_MLIR_TRANSFORMS_PASSES
#define PGEX_MLIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def MapToDialects : Pass<"map-to-dialects", "::mlir::ModuleOp"> {
  let summary = "Map functions to dialects.";
  let constructor = "pgex::createMapToDialectsPass()";
}

def SimplifyDataflow : FunctionPass<"simplify-dataflow"> {
  let summary = "Simplify the dataflow around bare tensor pointers.";
  let constructor = "pgex::createSimplifyDataflowPass()";
}

#endif
