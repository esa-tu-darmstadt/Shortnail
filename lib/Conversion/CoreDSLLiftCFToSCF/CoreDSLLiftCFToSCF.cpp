#include "shortnail/Conversion/Passes.h"

#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"

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

  void runOnOperation() override {
    auto isaxOp = getOperation();
    auto res =
        isaxOp->walk<WalkOrder::PreOrder>([&](coredsl::InstructionOp instr) {
          ControlFlowToSCFTransformation transformation;
          DominanceInfo dominance_info;
          LogicalResult res = transformCFGToSCF(instr.getRegion(),
                                                transformation, dominance_info);
          if (res.failed()) {
            instr->emitError("Failed to convert all cf ops to scf");
            return WalkResult::interrupt();
          }
          return WalkResult::skip();
        });
    res = isaxOp->walk<WalkOrder::PreOrder>([&](coredsl::AlwaysOp always) {
      ControlFlowToSCFTransformation transformation;
      DominanceInfo dominance_info;
      LogicalResult res =
          transformCFGToSCF(always.getRegion(), transformation, dominance_info);
      if (res.failed()) {
        always->emitError("Failed to convert all cf ops to scf");
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    });
    res = isaxOp->walk<WalkOrder::PostOrder>([&](coredsl::SpawnOp spawn) {
      ControlFlowToSCFTransformation transformation;
      DominanceInfo dominance_info;
      LogicalResult res =
          transformCFGToSCF(spawn.getRegion(), transformation, dominance_info);
      if (res.failed()) {
        spawn->emitError("Failed to convert all cf ops to scf");
        return WalkResult::interrupt();
      }
      return WalkResult::skip();
    });
    // TODO: are there some changes we need to notify other passes of
  }
};

} // anonymous namespace
