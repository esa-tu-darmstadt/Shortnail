//===- MergeISAX.cpp - Combine multiple ISAXes ----------------------------===//
//
// Copyright 2024 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#include "../PassDetail.h"

#include "shortnail/Conversion/MergeISAX.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "circt/Support/LLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"

// prefix to prepend to state elements
// for non-shared elements, an index will be appended to the prefix to avoid
// name conflicts
#define PREFIX "MERGED"

using namespace mlir;
using namespace mlir::shortnail;
using namespace mlir::coredsl;
using namespace circt;

namespace {
struct MergeISAX : public MergeIsaxBase<MergeISAX> {
  void runOnOperation() override;
};
} // namespace

// checks if all operations in a list are of similar type
template <typename T>
static void checkTypeEquality(SmallDenseMap<coredsl::ISAXOp, T> &regList) {
  for (auto [modOp, op] : regList) {
    auto referenceType = regList.begin()->second.getElementType();
    auto referenceConst = regList.begin()->second.getIsConst();
    auto referenceVolatile = regList.begin()->second.getIsVolatile();
    assert(op.getElementType() == referenceType);
    assert(op.getIsConst() == referenceConst);
    assert(op.getIsVolatile() == referenceVolatile);

    // additionally check element number for registers
    if constexpr (std::is_same_v<RegisterOp, T>) {
      auto referenceSize = regList.begin()->second.getNumElements();
      auto regOp = cast<RegisterOp>(op);
      assert(regOp.getNumElements() == referenceSize);
    }
  }
}

// finds all uses of an operation and replaces references with a new symbol
static void remapSymbolUses(Operation *oldOp, StringRef newSymbol) {
  auto *parent = oldOp->getParentWithTrait<OpTrait::SymbolTable>();
  auto uses = cast<SymbolOpInterface>(oldOp).getSymbolUses(parent);
  for (auto use : *uses) {
    TypeSwitch<Operation *>(use.getUser())
        .Case<GetOp>([&](auto g) { g.setSym(newSymbol); })
        .Case<SetOp>([&](auto g) { g.setSym(newSymbol); })
        .Case<AliasOp>([&](auto g) { g.setRef(newSymbol); })
        .Case<func::CallOp>([&](auto call) { call.setCallee(newSymbol); });
  }
}

// cloning of non-shared context, e.g. custom registers
template <typename T>
static void
cloneAndRemapOperations(SmallDenseMap<coredsl::ISAXOp, SmallVector<T>> &opMap,
                        SmallVectorImpl<coredsl::ISAXOp> &isaxes, OpBuilder &b) {
  // count ISAXes for prefix generation
  for (auto [isaxCtr, isax] : enumerate(isaxes)) {
    auto prefix = Twine(PREFIX) + Twine(isaxCtr);
    for (auto reg : opMap[isax]) {
      Operation *newRegOp = b.clone(*reg);
      // rename element (add prefix with isax index to avoid name conflicts)
      auto typedNewOp = cast<T>(newRegOp);
      typedNewOp.setSymNameAttr(
          b.getStringAttr(prefix + Twine(typedNewOp.getSymName())));
      remapSymbolUses(reg, typedNewOp.getSymName());
    }
  }
}

// cloning of shared context, e.g. arch refs/pc/fp regs/...
template <typename T>
static void cloneAndRemapOperations(SmallDenseMap<coredsl::ISAXOp, T> &regMap,
                                    OpBuilder &b) {
  // if any ISAX uses that state element
  if (regMap.size() > 0) {
    // clone the first element of the list
    // (since they are all equal)
    Operation *newRegOp = b.clone(*regMap.begin()->second);
    // add a prefix to avoid name conflicts
    auto typedNewOp = cast<T>(newRegOp);
    typedNewOp.setSymNameAttr(
        b.getStringAttr(Twine(PREFIX) + Twine(typedNewOp.getSymName())));

    // remap all uses of the old symbol names to the new shared one
    for (auto [isax, op] : regMap)
      remapSymbolUses(op, typedNewOp.getSymName());
  }
}

void MergeISAX::runOnOperation() {

  ModuleOp module = getOperation();

  // collect all registers per ISAX
  SmallDenseMap<coredsl::ISAXOp, RegisterOp> archRegsPerISAX, pcPerIsax,
      fpRegsPerISAX;
  SmallDenseMap<coredsl::ISAXOp, SmallVector<RegisterOp>> customRegsPerIsax;

  // collect aliasses per ISAX
  SmallDenseMap<coredsl::ISAXOp, SmallVector<AliasOp>> aliasPerIsax;

  // collect all functions per ISAX
  SmallDenseMap<coredsl::ISAXOp, SmallVector<func::FuncOp>> funcsPerISAX;

  // collect all addr spaces per ISAX
  SmallDenseMap<coredsl::ISAXOp, AddressSpaceOp> dmemPerISAX;

  // build an index of modules
  SmallVector<coredsl::ISAXOp> isaxes;

  //
  // collect state elements
  //

  // iterate over ISAXes
  for (auto op : module.getOps<coredsl::ISAXOp>()) {
    isaxes.push_back(op);

    // create vector for custom regs and aliases since an ISAX may contain >1
    auto &customRegsCurrentISAX = customRegsPerIsax[op];
    auto &aliasCurrentISAX = aliasPerIsax[op];
    auto &funcsCurrentISAX = funcsPerISAX[op];

    for (auto &wop : op.getOps()) {
      bool success =
          TypeSwitch<Operation *, bool>(&wop)
              .Case<RegisterOp>([&](RegisterOp rop) {
                switch (rop.getAccessMode()) {
                // arch reg file
                case RegisterAccessMode::core_x:
                  assert(!archRegsPerISAX.contains(op));
                  archRegsPerISAX[op] = rop;
                  break;
                // PC register
                case RegisterAccessMode::core_pc:
                  assert(!pcPerIsax.contains(op));
                  pcPerIsax[op] = rop;
                  break;
                // fp reg file
                case RegisterAccessMode::core_fp:
                  assert(!fpRegsPerISAX.contains(op));
                  fpRegsPerISAX[op] = rop;
                  break;
                // custom registers
                case RegisterAccessMode::local:
                  customRegsCurrentISAX.push_back(rop);
                  break;
                }
                return true;
              })
              .Case<AliasOp>([&](AliasOp aop) {
                // aliases
                aliasCurrentISAX.push_back(aop);
                return true;
              })
              .Case<func::FuncOp>([&](auto fun) {
                // aliases
                funcsCurrentISAX.push_back(fun);
                return true;
              })
              .Case<AddressSpaceOp>([&](AddressSpaceOp aop) {
                switch (aop.getAccessMode()) {
                // data memory
                case AddressSpaceAccessMode::core_mem:
                  assert(!dmemPerISAX.contains(op));
                  dmemPerISAX[op] = aop;
                  break;
                // error
                default:
                  op->emitError(
                      "Merging of address spaces other than data memory is "
                      "currently unsupported");
                  return false;
                }
                return true;
              })
              .Default([](Operation *op) { return true; });
      if (!success)
        return signalPassFailure();
    }
  }

  if (isaxes.size() == 1)
    return; // Nothing to merge we only have a single ISAX

  // make sure all architrectural state elements are of equal type
  checkTypeEquality(archRegsPerISAX);
  checkTypeEquality(pcPerIsax);
  checkTypeEquality(fpRegsPerISAX);
  checkTypeEquality(dmemPerISAX);

  //
  // Use collected information to move state elements and instructions
  //

  OpBuilder b{&getContext()};
  b.setInsertionPointToEnd(module.getBody());
  // rename top-level module to enable unified testing
  auto topIsax =
      coredsl::ISAXOp::create(b, module->getLoc(), b.getStringAttr("merged"));
  // Create a new empty block
  b.createBlock(&topIsax.getBody());
  b.setInsertionPointToEnd(&topIsax.getBody().front());

  // map state elements
  for (auto regMap : {archRegsPerISAX, fpRegsPerISAX, pcPerIsax}) {
    cloneAndRemapOperations(regMap, b);
  }
  cloneAndRemapOperations(dmemPerISAX, b);
  cloneAndRemapOperations(customRegsPerIsax, isaxes, b);
  cloneAndRemapOperations(aliasPerIsax, isaxes, b);
  cloneAndRemapOperations(funcsPerISAX, isaxes, b);

  // copy all blocks except registers, memory spaces and aliasses
  for (auto isax : isaxes) {
    for (auto &op : isax.getOps()) {

      // Ignore already cloned and remapped operations
      if (isa<RegisterOp, AliasOp, AddressSpaceOp, func::FuncOp>(op))
        continue;

      b.clone(op);
    }
    // remove old version of isax
    isax->erase();
  }
}

std::unique_ptr<Pass> shortnail::createMergeISAXPass() {
  return std::make_unique<MergeISAX>();
}
