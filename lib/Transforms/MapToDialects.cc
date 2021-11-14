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

/// Get the source tensor register.
static mlir::Value getLoadedTensor(CallOp caller, unsigned index,
                                   const PointerToTensor &ptr2tsr) {
  auto tv = caller.getOperand(index);

  auto load = tv.getDefiningOp<LLVM::LoadOp>();
  assert(load);

  auto ptr = load.addr();
  assert(ptr);
  assert(ptr2tsr.count(ptr));

  return ptr2tsr.lookup(ptr);
}

static void storeTensor(CallOp caller, unsigned index, mlir::Value tsr,
                        PointerToTensor &ptr2tsr) {
  auto tv = caller.getResult(index);
  // This is ensured by -simplify-dataflow
  assert(tv.hasOneUse()); // should be a llvm.store
  auto store = dyn_cast<LLVM::StoreOp>(*tv.user_begin());
  assert(store);
  auto ptr = store.addr();
  assert(ptr.getDefiningOp<LLVM::AllocaOp>());
  ptr2tsr.insert({ptr, tsr});
}

static void createTensorLoad(CallOp caller, PointerToTensor &ptr2tsr,
                             OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  b.setInsertionPointAfter(caller);
  auto tsr = b.create<memref::TensorLoadOp>(loc, caller.getOperand(0));

  storeTensor(caller, 0, tsr, ptr2tsr);
}

static void createTensorStore(CallOp caller, const PointerToTensor &ptr2tsr,
                              OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  assert(caller.getNumOperands() == 2);
  auto tsr = getLoadedTensor(caller, 0, ptr2tsr);

  b.setInsertionPointAfter(caller);
  b.create<memref::TensorStoreOp>(loc, tsr, caller.getOperand(1));
}

static void createTOSATensorAdd(CallOp caller, PointerToTensor &ptr2tsr,
                                OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  assert(caller.getNumOperands() == 2);
  auto tsrLHS = getLoadedTensor(caller, 0, ptr2tsr);
  auto tsrRHS = getLoadedTensor(caller, 1, ptr2tsr);

  b.setInsertionPointAfter(caller);
  auto tsr = b.create<tosa::AddOp>(loc, tsrLHS.getType(), tsrLHS, tsrRHS);

  storeTensor(caller, 0, tsr, ptr2tsr);
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

static mlir::ArrayAttr getArrayAttr(ArrayRef<int64_t> arr, OpBuilder &b) {
  SmallVector<mlir::Attribute> attrs;
  transform(arr, std::back_inserter(attrs), [&](int64_t i) {
    return b.getIntegerAttr(b.getIntegerType(64), i);
  });
  return b.getArrayAttr(attrs);
}

static void createTOSATensorReshape(CallOp caller, PointerToTensor &ptr2tsr,
                                    OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  // Resolve input tensor pointer
  assert(caller.getNumOperands() >= 2);
  auto tsr = getLoadedTensor(caller, 0, ptr2tsr);

  // Resolve attributes
  SmallVector<int64_t> shape;
  getIntegerArray(caller, shape, 1, caller.getNumOperands() - 1);

  auto ty = tsr.getType().cast<TensorType>();
  auto newTy = ty.clone(llvm::makeArrayRef(shape));

  b.setInsertionPointAfter(caller);
  auto reshape =
      b.create<tosa::ReshapeOp>(loc, newTy, tsr, getArrayAttr(shape, b));
  storeTensor(caller, 0, reshape.getResult(), ptr2tsr);
}

static void createTOSAConv2d(CallOp caller, PointerToTensor &ptr2tsr,
                             OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  Location loc = caller.getLoc();

  auto input = getLoadedTensor(caller, 0, ptr2tsr);
  auto weight = getLoadedTensor(caller, 1, ptr2tsr);
  auto bias = getLoadedTensor(caller, 2, ptr2tsr);

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
  storeTensor(caller, 0, conv2d.getResult(), ptr2tsr);
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
      if (name.find("tensor_load") != StringRef::npos)
        createTensorLoad(caller, ptr2tsr, b);
      else if (name.find("tensor_store") != StringRef::npos)
        createTensorStore(caller, ptr2tsr, b);
      else if (name == "tensor_add")
        createTOSATensorAdd(caller, ptr2tsr, b);
      else if (name.find("tensor_reshape") != StringRef::npos)
        createTOSATensorReshape(caller, ptr2tsr, b);
      else if (name == "tensor_conv2d")
        createTOSAConv2d(caller, ptr2tsr, b);

      toErase.push_back(caller);
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
