#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/Debug.h"

#include <memory>

using namespace mlir;
using namespace llvm;
using namespace pgex;

#define PASS_NAME "map-to-memref"
#define DEBUG_TYPE PASS_NAME

using PointerToTensor = llvm::DenseMap<mlir::Value, mlir::Value>;

static void createTensorLoad(CallOp caller, PointerToTensor &ptr2tsr,
                             OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  // -- step 1: find the tensor object (the llvm.ptr<ptr<i8>> value)
  assert(caller.getNumResults() == 1);

  auto tv = caller.getResult(0);
  assert(tv.hasOneUse()); // should be a llvm.store

  auto store = dyn_cast<LLVM::StoreOp>(*tv.user_begin());
  assert(store);

  auto ptr = store.addr();
  assert(ptr.getDefiningOp<LLVM::AllocaOp>());

  // -- step 2: build the tensor mapping.
  b.setInsertionPointAfter(caller);
  auto tsr = b.create<memref::TensorLoadOp>(loc, caller.getOperand(0));
  ptr2tsr.insert({ptr, tsr});
}

static void createTensorStore(CallOp caller, const PointerToTensor &ptr2tsr,
                              OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  assert(caller.getNumOperands() == 2);
  auto tv = caller.getOperand(0);

  auto load = tv.getDefiningOp<LLVM::LoadOp>();
  assert(load);

  auto ptr = load.addr();
  assert(ptr);
  assert(ptr2tsr.count(ptr));

  auto tsr = ptr2tsr.lookup(ptr);

  b.setInsertionPointAfter(caller);
  b.create<memref::TensorStoreOp>(loc, tsr, caller.getOperand(1));
}

static void createTOSATensorAdd(CallOp caller, PointerToTensor &ptr2tsr,
                                OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  assert(caller.getNumOperands() == 2);

  // Mapping operands
  auto lhs = caller.getOperand(0);
  auto rhs = caller.getOperand(1);

  auto loadLHS = lhs.getDefiningOp<LLVM::LoadOp>();
  auto loadRHS = rhs.getDefiningOp<LLVM::LoadOp>();
  assert(loadLHS && loadRHS);

  auto ptrLHS = loadLHS.addr();
  auto ptrRHS = loadRHS.addr();
  assert(ptr2tsr.count(ptrLHS) && ptr2tsr.count(ptrRHS));

  auto tsrLHS = ptr2tsr.lookup(ptrLHS);
  auto tsrRHS = ptr2tsr.lookup(ptrRHS);

  b.setInsertionPointAfter(caller);
  auto tsr = b.create<tosa::AddOp>(loc, tsrLHS.getType(), tsrLHS, tsrRHS);

  // Mapping result
  assert(caller.getNumResults() == 1);

  auto tv = caller.getResult(0);
  assert(tv.hasOneUse()); // should be a llvm.store

  auto store = dyn_cast<LLVM::StoreOp>(*tv.user_begin());
  assert(store);

  auto ptr = store.addr();
  assert(ptr.getDefiningOp<LLVM::AllocaOp>());
  ptr2tsr.insert({ptr, tsr});
}

static void cleanUp(ModuleOp m, ArrayRef<Operation *> toErase,
                    PointerToTensor &ptr2tsr) {
  // Erase callers
  llvm::SetVector<llvm::StringRef> names;
  for (auto op : toErase) {
    if (auto caller = dyn_cast<CallOp>(op))
      names.insert(caller.getCallee());
    for (auto user : op->getUsers())
      if (user->use_empty())
        user->erase();
    if (op->use_empty())
      op->erase();
  }

  // Erase callees
  m.walk([&](FuncOp f) {
    if (names.count(f.getName()))
      f.erase();
  });

  // Erase allocated tensors
  for (auto &it : ptr2tsr) {
    auto ptr = it.first;
    for (auto user : ptr.getUsers())
      if (user->use_empty())
        user->erase();
    if (ptr.use_empty())
      ptr.getDefiningOp()->erase();
  }
}

namespace {
struct MapToDialectsPass : public ::pgex::MapToDialectsBase<MapToDialectsPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder b(m.getContext());

    // From pointer values to tensors.
    llvm::DenseMap<mlir::Value, mlir::Value> ptr2tsr;
    SmallVector<Operation *> toErase;

    // Create memref.tensor_load and tensor_store.
    m.walk([&](CallOp caller) {
      auto name = caller.getCallee();
      if (name.find("tensor_load") != StringRef::npos) {
        createTensorLoad(caller, ptr2tsr, b);
        toErase.push_back(caller);
      } else if (name.find("tensor_store") != StringRef::npos) {
        createTensorStore(caller, ptr2tsr, b);
        toErase.push_back(caller); // store can be directly removed.
      } else if (name == "tensor_add") {
        createTOSATensorAdd(caller, ptr2tsr, b);
        toErase.push_back(caller);
      }
    });

    // Clean up
    cleanUp(m, toErase, ptr2tsr);
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
pgex::createMapToDialectsPass() {
  return std::make_unique<MapToDialectsPass>();
}
