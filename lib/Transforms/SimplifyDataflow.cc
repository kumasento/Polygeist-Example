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

#define PASS_NAME "simplify-dataflow"
#define DEBUG_TYPE PASS_NAME

static bool isCalleeTensor(CallOp caller) {
  return caller.getCallee().find("tensor_") != StringRef::npos;
}

static bool isValueLLVMPtrI8(mlir::Value value) {
  if (auto ptrTy = value.getType().dyn_cast<LLVM::LLVMPointerType>()) {
    return ptrTy.getElementType().isInteger(8);
  }
  return false;
}

static void getTensorPtrRegisters(FuncOp f,
                                  llvm::SetVector<mlir::Value> &ptrs) {
  f.walk([&](CallOp caller) {
    if (isCalleeTensor(caller)) {
      // Load a tensor pointer from a target register.
      for (auto operand : caller->getOperands())
        if (auto loadOp = operand.getDefiningOp<LLVM::LoadOp>())
          if (isValueLLVMPtrI8(operand))
            ptrs.insert(loadOp.addr());

      // Store a tensor pointer to a target register.
      for (auto result : caller->getResults())
        for (auto user : result.getUsers())
          if (auto storeOp = dyn_cast<LLVM::StoreOp>(user))
            if (isValueLLVMPtrI8(storeOp.value()))
              ptrs.insert(storeOp.addr());
    }
  });
}

static mlir::Value
findMultiStoreRegister(FuncOp f, const llvm::SetVector<mlir::Value> &ptrs) {
  for (auto ptr : ptrs) {
    bool hasStore = false;
    for (auto user : ptr.getUsers())
      if (isa<LLVM::StoreOp>(user)) {
        if (hasStore)
          return ptr;
        hasStore = true;
      }
  }

  return nullptr;
}

static Operation *findLastStore(FuncOp f, mlir::Value ptr) {
  DominanceInfo dom(f);
  SmallVector<Operation *> storeOps;
  for (auto user : ptr.getUsers())
    if (isa<LLVM::StoreOp>(user))
      storeOps.push_back(user);

  for (auto fst : storeOps) {
    bool dominateOthers = false;
    for (auto snd : storeOps) {
      if (snd != fst && dom.dominates(fst, snd)) {
        dominateOthers = true;
        break;
      }
    }
    if (!dominateOthers)
      return fst;
  }
  return nullptr;
}

static void duplicateMultiStore(FuncOp f, llvm::SetVector<mlir::Value> &ptrs) {
  OpBuilder b(f.getContext());

  while (auto ptr = findMultiStoreRegister(f, ptrs)) {
    Operation *lastStore = findLastStore(f, ptr);
    if (!lastStore)
      return;

    b.setInsertionPointAfterValue(ptr);
    auto cloned = b.clone(*ptr.getDefiningOp());
    auto newPtr = cloned->getResult(0);
    ptrs.insert(newPtr);

    lastStore->setOperand(1, newPtr);

    DominanceInfo dom(f);
    ptr.replaceUsesWithIf(newPtr, [&](OpOperand &operand) {
      return dom.dominates(lastStore, operand.getOwner());
    });
  }
}

static void bufferizeImmediateResults(FuncOp f,
                                      llvm::SetVector<mlir::Value> &ptrs) {
  OpBuilder b(f.getContext());

  mlir::Value dummy = nullptr;
  if (ptrs.empty()) {
    b.setInsertionPointToStart(&(f.getBlocks().front()));
    Location loc = f.getLoc();
    mlir::Value c1 = b.create<arith::ConstantIntOp>(loc, 1, 64);

    mlir::Value ptr = b.create<LLVM::AllocaOp>(
        loc,
        LLVM::LLVMPointerType::get(
            LLVM::LLVMPointerType::get(b.getIntegerType(8))),
        c1);
    dummy = ptr;
    ptrs.insert(ptr); // just a placeholder
  }

  f.walk([&](CallOp caller) {
    if (isCalleeTensor(caller)) {
      for (unsigned i = 0; i < caller.getNumOperands(); ++i) {
        auto operand = caller.getOperand(i);
        if (!isValueLLVMPtrI8(operand))
          continue;
        if (operand.getDefiningOp<LLVM::LoadOp>())
          continue;

        // Need to bufferize
        if (auto srcCaller = operand.getDefiningOp<CallOp>()) {
          if (isCalleeTensor(srcCaller)) {
            Location loc = srcCaller.getLoc();

            b.setInsertionPointAfterValue(ptrs.front());
            auto cloned = b.clone(*ptrs.front().getDefiningOp());
            auto ptr = cloned->getResult(0);

            b.setInsertionPointAfter(srcCaller);
            b.create<LLVM::StoreOp>(loc, operand, ptr);

            b.setInsertionPoint(caller);
            auto loadOp = b.create<LLVM::LoadOp>(loc, ptr);
            caller->setOperand(i, loadOp.res());
          }
        }
      }
    }
  });

  if (dummy)
    dummy.getDefiningOp()->erase();
}

static bool findParentOp(Operation *op, Operation *parent) {
  if (!op)
    return false;
  if (op == parent)
    return true;
  if (isa<FuncOp>(op))
    return false;
  return findParentOp(op->getParentOp(), parent);
}

/// Make sure there is no cross region reference to the same register.
static void reg2Mem(Block *block, llvm::SetVector<mlir::Value> &ptrs,
                    llvm::MapVector<mlir::Value, mlir::Operation *> &scopes,
                    llvm::SetVector<mlir::Operation *> &ignore) {
  OpBuilder b(block->getParentOp()->getContext());

  auto loadFrom = [&](const mlir::Value &src, const mlir::Value &dst,
                      const Location loc) {
    auto val = b.create<LLVM::LoadOp>(loc, src);
    ignore.insert(val.getOperation());
    ignore.insert(b.create<LLVM::StoreOp>(loc, val, dst));
  };

  for (auto &op : *block) {
    if (isa<LLVM::LoadOp, LLVM::StoreOp>(&op) && !ignore.count(&op)) {
      auto ptr = isa<LLVM::LoadOp>(&op) ? op.getOperand(0) : op.getOperand(1);
      if (!ptrs.count(ptr)) // The load/store value is not a valid tensor.
        continue;

      if (scopes.lookup(ptr) != op.getParentOp()) {
        // Not defined within the same block.
        b.setInsertionPointAfter(ptr.getDefiningOp());

        // Cloned new ptr.
        auto cln = b.clone(*ptr.getDefiningOp())->getResult(0);
        ptrs.insert(cln); // should track it within the ptrs.
        scopes.insert({cln, op.getParentOp()});

        // Replace all the uses within the current block.
        ptr.replaceUsesWithIf(cln, [&](OpOperand &operand) -> bool {
          return findParentOp(operand.getOwner(), op.getParentOp());
        });

        // Initialize the register.
        b.setInsertionPoint(op.getParentOp());
        loadFrom(ptr, cln, op.getLoc());

        // Store the result back.
        b.setInsertionPointAfter(op.getParentOp());
        loadFrom(cln, ptr, op.getLoc());
      }
    } else {
      for (Region &region : op.getRegions())
        for (Block &blk : region)
          reg2Mem(&blk, ptrs, scopes, ignore);
    }
  }
}

namespace {
struct SimplifyDataflowPass
    : public ::pgex::SimplifyDataflowBase<SimplifyDataflowPass> {
  void runOnFunction() override {
    FuncOp f = getFunction();

    llvm::SetVector<mlir::Value> ptrs;
    getTensorPtrRegisters(f, ptrs);
    LLVM_DEBUG({
      dbgs() << "Found registers:\n";
      for (auto ptr : ptrs) {
        dbgs() << ptr << '\n';
      }
    });

    // ---  Step 0: find all tensor pointers through the operands and results
    // from tensor callers.
    if (reg2mem) {
      llvm::MapVector<mlir::Value, mlir::Operation *> scopes;

      for (auto ptr : ptrs)
        scopes.insert({ptr, ptr.getDefiningOp()->getParentOp()});

      // --- Step 1: reg2mem.
      llvm::SetVector<mlir::Operation *> ignore;
      for (Block &block : f.getBlocks())
        reg2Mem(&block, ptrs, scopes, ignore);
    }

    if (duplicate) {
      LLVM_DEBUG(dbgs() << "Duplicate multi store.\n");
      // --- Step 2: if a register has been store multiple times, we replicate
      // each of them.
      duplicateMultiStore(f, ptrs);
    }

    // --- Step 3: insert a register and corresponding load store for an
    // immediate value from a tensor function call.
    bufferizeImmediateResults(f, ptrs);
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
pgex::createSimplifyDataflowPass() {
  return std::make_unique<SimplifyDataflowPass>();
}
