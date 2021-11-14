#include "pgex/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace pgex;

int main(int argc, char *argv[]) {
  DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<StandardOpsDialect>();
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::arith::ArithmeticDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<mlir::tosa::TosaDialect>();

  // Register the standard passes we want.
  registerCanonicalizerPass();
  registerCSEPass();
  registerInlinerPass();

  registerMapToDialectsPass();
  registerSimplifyDataflowPass();

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  // Register printer command line options.
  registerAsmPrinterCLOptions();

  return failed(MlirOptMain(argc, argv, "Phism optimizer driver", registry,
                            /*preloadDialectsInContext=*/true));
}
