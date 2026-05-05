#include "shortnail/Conversion/Passes.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace shortnail {
#define GEN_PASS_DEF_COREDSLLIFTCFTOSCF
#include "shortnail/Conversion/Passes.h.inc"
} // namespace shortnail
} // namespace mlir

using namespace mlir;
using namespace mlir::shortnail;

namespace {

struct CoreDSLLiftCFToSCF
    : public mlir::shortnail::impl::CoreDSLLiftCFToSCFBase<CoreDSLLiftCFToSCF> {
  using CoreDSLLiftCFToSCFBase::CoreDSLLiftCFToSCFBase;

  // This needs to be a class rather than a lambda because we cannot have
  // templated lambdas and this needs to be explicitly instantiated with
  // multiple op types for the walk method
  template <typename OpType>
  struct RegionToSCFConverter {
    // So we can use Pass::getChildAnalysis
    friend struct CoreDSLLiftCFToSCF;
    CoreDSLLiftCFToSCF &pass;
    bool &changed;

    WalkResult operator()(OpType op) {
      ControlFlowToSCFTransformation transformation;
      DominanceInfo &dominanceInfo = pass.getChildAnalysis<DominanceInfo>(op);
      FailureOr<bool> res = transformCFGToSCF(op.getRegion(), transformation, dominanceInfo);
      if (failed(res)) {
        op->emitError("Failed to convert all cf ops to scf");
        return WalkResult::interrupt();
      }
      changed |= *res;
      return WalkResult::skip();
    }
  };

  void runOnOperation() override {
    auto isaxOp = getOperation();
    // Visit all operations that can have multiple child blocks. As SCF ops
    // other than scf::ExecuteRegionOp cannot have multiple child blocks, they
    // don't need to be visited here
    bool changed = false;
    static constexpr WalkOrder ORDER = WalkOrder::PostOrder;
    auto res = isaxOp->walk<ORDER>(RegionToSCFConverter<coredsl::InstructionOp>{*this, changed});
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    res = isaxOp->walk<ORDER>(RegionToSCFConverter<coredsl::AlwaysOp>{*this, changed});
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    res = isaxOp->walk<ORDER>(RegionToSCFConverter<coredsl::SpawnOp>{*this, changed});
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    res = isaxOp->walk<ORDER>(RegionToSCFConverter<func::FuncOp>{*this, changed});
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    // TODO: this will always be a child of the other ops, so we could just do this recursively in the callback
    res = isaxOp->walk<ORDER>(RegionToSCFConverter<scf::ExecuteRegionOp>{*this, changed});
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};

} // anonymous namespace
