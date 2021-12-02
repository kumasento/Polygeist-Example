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
  let options = [
    Option<"reg2mem", "reg2mem", "bool", /*default=*/"false", "run reg2mem">,
    Option<"duplicate", "duplicate", "bool", /*default=*/"false", "run duplicate multiple store">,
  ];
}

def Ptr2Tsr: Pass<"ptr2tsr", "::mlir::ModuleOp"> {
  let summary = "From LLVM pointers to tensors.";
  let constructor = "pgex::createPtr2TsrPass()";
}

#endif
