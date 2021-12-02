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

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <memory>

using namespace mlir;
using namespace llvm;
using namespace pgex;

#define PASS_NAME "ptr2tsr"
#define DEBUG_TYPE PASS_NAME

static mlir::ArrayAttr getArrayAttr(ArrayRef<int64_t> arr, OpBuilder &b) {
  SmallVector<mlir::Attribute> attrs;
  transform(arr, std::back_inserter(attrs), [&](int64_t i) {
    return b.getIntegerAttr(b.getIntegerType(64), i);
  });
  return b.getArrayAttr(attrs);
}

static void getIntegerArray(CallOp caller, SmallVectorImpl<int64_t> &arr,
                            unsigned begin, unsigned size) {
  for (unsigned i = begin; i < begin + size; ++i) {
    auto operand = caller.getOperand(i);
    auto cst = operand.getDefiningOp<arith::ConstantIntOp>();
    assert(cst);
    arr.push_back(cst.value());
  }
}

static mlir::Value createTOSATensorReshape(CallOp caller, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  // Resolve input tensor pointer
  assert(caller.getNumResults() == 1);
  assert(caller.getNumOperands() >= 2);
  auto tsr = caller.getOperand(0);
  assert(tsr.getType().isa<TensorType>());

  // Resolve attributes
  SmallVector<int64_t> shape;
  getIntegerArray(caller, shape, 1, caller.getNumOperands() - 1);

  auto ty = tsr.getType().cast<TensorType>();
  auto newTy = ty.clone(llvm::makeArrayRef(shape));

  b.setInsertionPointAfter(caller);
  auto reshape =
      b.create<tosa::ReshapeOp>(loc, newTy, tsr, getArrayAttr(shape, b));
  caller.getResult(0).replaceAllUsesWith(reshape.getResult());

  return reshape.getResult();
}

static mlir::Value createTOSATensorAdd(CallOp caller, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  assert(caller.getNumOperands() == 2);
  auto lhs = caller.getOperand(0);
  auto rhs = caller.getOperand(1);

  b.setInsertionPointAfter(caller);
  auto tsr = b.create<tosa::AddOp>(loc, lhs.getType(), lhs, rhs);
  caller.getResult(0).replaceAllUsesWith(tsr);

  return tsr;
}

static mlir::Value createTOSAConv2d(CallOp caller, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  assert(caller.getNumOperands() == 11);
  auto input = caller.getOperand(0);
  auto weight = caller.getOperand(1);
  auto bias = caller.getOperand(2);

  SmallVector<int64_t> pad, stride, dilation;
  getIntegerArray(caller, pad, 3, 4);
  getIntegerArray(caller, stride, 7, 2);
  getIntegerArray(caller, dilation, 9, 2);

  auto ty = input.getType().cast<TensorType>();
  auto retTy = UnrankedTensorType::get(ty.getElementType());

  b.setInsertionPointAfter(caller);
  auto conv2d = b.create<tosa::Conv2DOp>(
      loc, retTy, input, weight, bias, getArrayAttr(pad, b),
      getArrayAttr(stride, b), getArrayAttr(dilation, b));
  caller.getResult(0).replaceAllUsesWith(conv2d);
  return conv2d;
}

static mlir::Value createTOSAConcat(CallOp caller, OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  assert(caller.getNumOperands() >= 3);

  SmallVector<mlir::Value> operands{caller.getOperands().drop_front()};

  // If there is any unranked type, we use that as the output.
  auto it = find_if(caller.getOperands(), [&](mlir::Value value) {
    return value.getType().isa<UnrankedTensorType>();
  });

  auto ty = caller.getOperand(1).getType();
  if (it == caller.operand_end()) {
    // Make the shape dynamic.
    if (auto ranked = ty.dyn_cast<RankedTensorType>())
      ty = RankedTensorType::get(
          SmallVector<int64_t>(ranked.getShape().size(), -1),
          ranked.getElementType());
  } else {
    ty = UnrankedTensorType::get(ty.cast<TensorType>().getElementType());
  }

  b.setInsertionPointAfter(caller);
  auto concat = b.create<tosa::ConcatOp>(
      loc, ty, operands,
      b.getIntegerAttr(
          b.getIntegerType(64),
          caller.getOperand(0).getDefiningOp<arith::ConstantIntOp>().value()));

  caller.getResult(0).replaceAllUsesWith(concat);
  return concat;
}

static void createMemRefTensorStore(CallOp caller, OpBuilder &b) {
  b.setInsertionPointAfter(caller);

  auto getTensor = [&](mlir::Value tsr) -> mlir::Value {
    auto ty = tsr.getType();
    if (auto tsrTy = ty.dyn_cast<RankedTensorType>())
      if (tsrTy.getShape().size() == 1 && tsrTy.getShape()[0] == -1)
        return tsr;

    return b.create<tensor::CastOp>(
        caller.getLoc(),
        RankedTensorType::get({-1}, ty.cast<TensorType>().getElementType()),
        tsr);
  };

  b.create<memref::TensorStoreOp>(
      caller.getLoc(), getTensor(caller.getOperand(0)), caller.getOperand(1));
}

/// -----------------------------------------------------------------------

struct PtrInfo {
  LLVM::AllocaOp def;               // definition
  LLVM::StoreOp store;              // the first store
  CallOp caller;                    // caller to tensor_load
  bool start;                       // Is start ptr.
  memref::TensorLoadOp tensor_load; // newly created tensor_load.

  explicit PtrInfo(LLVM::AllocaOp def, LLVM::StoreOp store, CallOp caller)
      : def(def), store(store), caller(caller),
        start(caller.getCallee().find("tensor_load") != StringRef::npos) {}
};

struct Ptr2Tsr {
  FuncOp f;                             // top function
  SmallVector<PtrInfo> ptrs;            // llvm.alloca defined tensors.
  llvm::SetVector<Operation *> toErase; // funcs to erase

  explicit Ptr2Tsr(FuncOp f) : f(f) {}

  void initPtrs();
  void process(PtrInfo &);
  void postProcess();
  void run();
  mlir::Value replace(CallOp, mlir::Value);
};

void Ptr2Tsr::initPtrs() {
  ptrs.clear();

  OpBuilder b(f.getContext());
  DominanceInfo dom(f);

  f.walk([&](LLVM::AllocaOp alloca) {
    auto users = alloca.getResult().getUsers();
    if (any_of(users, [&](const Operation *user) {
          return !isa<LLVM::LoadOp, LLVM::StoreOp>(user);
        }))
      return;

    SmallVector<Operation *> storeOps;
    copy_if(users, std::back_inserter(storeOps),
            [&](const Operation *user) { return isa<LLVM::StoreOp>(user); });

    // Within all store ops, find the one that dominates all the other users.
    auto it = find_if(storeOps, [&](Operation *store) {
      return all_of(users, [&](Operation *user) {
        return dom.dominates(store, user) || user == store;
      });
    });
    if (it == storeOps.end())
      return;

    auto store = dyn_cast_or_null<LLVM::StoreOp>(*it);
    if (!store)
      return;

    // Check if the value to be stored is defined by a tensor_load function.
    auto toStore = store.value();
    if (toStore.isa<BlockArgument>())
      return;
    auto caller = toStore.getDefiningOp<CallOp>();
    if (!caller)
      return;

    ptrs.emplace_back(alloca, store, caller);
  });

  // Create tensors
  for (auto &ptr : ptrs) {
    if (ptr.start) {
      b.setInsertionPointAfter(ptr.caller);
      ptr.tensor_load = b.create<memref::TensorLoadOp>(
          ptr.caller.getLoc(), ptr.caller.getOperand(0));
    }
  }
}

mlir::Value Ptr2Tsr::replace(CallOp caller, mlir::Value target) {
  OpBuilder b(f.getContext());

  auto name = caller.getCallee();
  if (name.find("tensor_reshape") != StringRef::npos)
    return createTOSATensorReshape(caller, b);
  if (name.find("tensor_add") != StringRef::npos)
    return createTOSATensorAdd(caller, b);
  if (name.find("tensor_conv2d") != StringRef::npos)
    return createTOSAConv2d(caller, b);
  if (name.find("tensor_concat") != StringRef::npos)
    return createTOSAConcat(caller, b);
  assert(false && "Cannot replace");
}

void Ptr2Tsr::process(PtrInfo &ptr) {
  auto users = ptr.def.getResult().getUsers();
  llvm::SetVector<Operation *> userSet{users.begin(), users.end()};

  auto tsr = ptr.tensor_load.getResult();
  OpBuilder b(f.getContext());

  std::function<mlir::Value(Block *, mlir::Value)> helper =
      [&](Block *block, mlir::Value entry) -> mlir::Value {
    bool hasLoad = false, hasStore = false;
    auto tsr = entry;
    for (Operation &op : *block) {
      // This op is a user. We try to replace the use of a loaded value by
      // the corresponding tensor, and change the function call if the
      // target is a store op. We only look at those operations related to
      // the current ptr, and if the ptr is a start ptr (defined by
      // tensor_load), we will check if the current op is the first store op
      // since that case has been resolved earlier.
      if (userSet.count(&op) && (!ptr.start || ptr.store != &op)) {
        if (auto load = dyn_cast<LLVM::LoadOp>(&op)) {
          hasLoad = true;
          load.getResult().replaceAllUsesWith(tsr);
        } else {
          hasStore = true;
          auto storeOp = dyn_cast<LLVM::StoreOp>(&op);
          assert(storeOp);

          auto defOp = storeOp.value().getDefiningOp();

          // The value to be stored could be defined by a caller to a tensor
          // function. In that case, we replace the function call with the
          // corresponding operation.
          if (auto caller = dyn_cast<CallOp>(defOp)) {
            // The caller has not yet been replaced.
            tsr = replace(caller, storeOp.value());
            caller.erase();
            // toErase.insert(caller);
          } else {
            llvm_unreachable("nOt implemented");
          }
        }
      } else if (op.getNumRegions() > 0) {
        hasLoad = hasStore = true;
        for (Region &region : op.getRegions())
          for (Block &blk : region)
            tsr = helper(&blk, tsr);
      }
    }

    // Promote the tensor to the outside if there is an update.
    if (auto forOp = dyn_cast<scf::ForOp>(block->getParentOp())) {
      if (hasLoad || hasStore) {
        // If the value to yield has a different type (in shape) with the entry,
        // we need to cast both to dynamic shape.
        if (entry.getType() != tsr.getType()) {
          assert(tsr.getType().isa<UnrankedTensorType>() ||
                 (tsr.getType().isa<RankedTensorType>() &&
                  !tsr.getType().cast<RankedTensorType>().hasStaticShape()));
          b.setInsertionPointAfterValue(entry);
          auto castOp =
              b.create<tensor::CastOp>(forOp.getLoc(), tsr.getType(), entry);
          entry.replaceUsesWithIf(castOp, [&](OpOperand &operand) {
            return operand.getOwner()->getBlock() == forOp.getBody();
          });
          entry = castOp;
        }

        // There is load and store, we need to promote the inner use of entry
        // into iter_args, and add the stored value to the yield result.
        SmallVector<mlir::Value> iterArgs = forOp.initArgs();
        iterArgs.push_back(entry);
        b.setInsertionPoint(forOp);
        auto newForOp =
            b.create<scf::ForOp>(forOp.getLoc(), forOp.lowerBound(),
                                 forOp.upperBound(), forOp.step(), iterArgs);

        b.setInsertionPointToStart(newForOp.getBody());
        BlockAndValueMapping bvm;
        for (Operation &op : *forOp.getBody()) {
          auto cloned = b.clone(op, bvm);
          if (toErase.count(&op))
            toErase.insert(cloned);
        }

        // Use the block argument.
        entry.replaceUsesWithIf(
            newForOp.getBody()->getArguments().back(), [&](OpOperand &operand) {
              return operand.getOwner()->getBlock() == newForOp.getBody();
            });

        // Pass in as iter_arg.
        Block *body = newForOp.getBody();
        auto yield = cast<scf::YieldOp>(body->getTerminator());
        b.setInsertionPointAfter(yield);
        SmallVector<mlir::Value> yieldOperands{yield.operand_begin(),
                                               yield.operand_end()};
        yieldOperands.push_back(bvm.lookup(tsr));

        b.create<scf::YieldOp>(newForOp.getLoc(), yieldOperands);

        yield.erase();

        // Update the result being passed later.
        tsr = newForOp.getResult(newForOp.getNumResults() - 1);
        toErase.insert(forOp);
      }
    }

    return tsr;
  };

  for (Block &block : f.getBlocks())
    tsr = helper(&block, tsr);
}

/// Replace those tensor function calls that doesn't have a returned value.
void Ptr2Tsr::postProcess() {
  OpBuilder b(f.getContext());

  f.walk([&](CallOp caller) {
    auto name = caller.getCallee();
    if (name.find("tensor_store") != StringRef::npos) {
      createMemRefTensorStore(caller, b);
      toErase.insert(caller);
    }
  });

  for (auto op : toErase) {
    op->erase();
  }

  for (auto &ptr : ptrs) {
    bool noUsers = true;
    for (auto user : ptr.def.getResult().getUsers()) {
      if (!user->use_empty())
        noUsers = false;
      else
        user->erase();
    }

    if (ptr.start)
      ptr.caller.erase();
    if (noUsers)
      ptr.def.erase();
  }
}

void Ptr2Tsr::run() {
  initPtrs();
  for (auto &ptr : ptrs)
    if (ptr.start)
      process(ptr);
  for (auto &ptr : ptrs)
    if (!ptr.start)
      process(ptr);
  postProcess();
}

static void process(FuncOp f) {
  // Step 1: find all the llvm.alloca that correspond to a tensor.
  // We assume that all tensors should start with a memref.tensor_load.
  Ptr2Tsr(f).run();
}

namespace {
struct Ptr2TsrPass : public ::pgex::Ptr2TsrBase<Ptr2TsrPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();

    m.walk([&](FuncOp f) {
      if (f.getBlocks().empty())
        return;

      process(f);
    });
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> pgex::createPtr2TsrPass() {
  return std::make_unique<Ptr2TsrPass>();
}
