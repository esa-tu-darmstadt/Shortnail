#include "shortnail/Conversion/Passes.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace shortnail {
#define GEN_PASS_DEF_COREDSLLIFTCFTOSCF
#include "shortnail/Conversion/Passes.h.inc"
} // namespace shortnail
} // namespace mlir

using namespace mlir;
using namespace mlir::shortnail;

namespace {

template <typename OpType>
WalkResult convertRegionToSCF(OpType op) {
  ControlFlowToSCFTransformation transformation;
  DominanceInfo dominanceInfo;
  LogicalResult res = transformCFGToSCF(op.getRegion(), transformation, dominanceInfo);
  if (res.failed()) {
    op->emitError("Failed to convert all cf ops to scf");
    return WalkResult::interrupt();
  }
  return WalkResult::skip();
}

struct CoreDSLLiftCFToSCF
    : public mlir::shortnail::impl::CoreDSLLiftCFToSCFBase<CoreDSLLiftCFToSCF> {
  using CoreDSLLiftCFToSCFBase::CoreDSLLiftCFToSCFBase;

  void runOnOperation() override {
    auto isaxOp = getOperation();
    // TODO: does this work for cf ops that are in deeper nesting?
    // We can't use the LiftControlFlowToSCF pass, because it only visits func
    // operations
    auto res = isaxOp->walk<WalkOrder::PreOrder>(convertRegionToSCF<coredsl::InstructionOp>);
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    res = isaxOp->walk<WalkOrder::PreOrder>(convertRegionToSCF<coredsl::AlwaysOp>);
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    res = isaxOp->walk<WalkOrder::PreOrder>(convertRegionToSCF<coredsl::SpawnOp>);
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    res = isaxOp->walk<WalkOrder::PreOrder>(convertRegionToSCF<func::FuncOp>);
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
  }
};

} // anonymous namespace
