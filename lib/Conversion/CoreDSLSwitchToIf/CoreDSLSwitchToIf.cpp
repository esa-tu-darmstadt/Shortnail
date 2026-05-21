#include "mlir/Transforms/DialectConversion.h"
#include "shortnail/Conversion/Passes.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

namespace mlir {
namespace shortnail {
#define GEN_PASS_DEF_COREDSLSWITCHTOIF
#include "shortnail/Conversion/Passes.h.inc"
} // namespace shortnail
} // namespace mlir

using namespace mlir;
using namespace mlir::shortnail;
using namespace circt;

namespace {

struct IndexSwitchToSCFIf : public OpConversionPattern<scf::IndexSwitchOp> {
  using OpConversionPattern<scf::IndexSwitchOp>::OpConversionPattern;

  scf::IfOp convertCases(ConversionPatternRewriter &rewriter,
                         scf::IndexSwitchOp op, unsigned caseIdx) const {
    assert(caseIdx < op.getNumCases());
    // NOTE: Each scf.if emitted will have the same location as the top level
    // switch
    const Location loc = op.getLoc();
    const int64_t caseVal = op.getCases()[caseIdx];
    auto arg = op.getArg();
    // We assume that no index ops other than the ones introduced by
    // ControlFlowToSCF exist, which means that this operation must be an
    // index_cast of the argument of the original cf.switch
    assert(isa<arith::IndexCastUIOp>(arg.getDefiningOp()) &&
           "ControlFlowToSCF pass should only generate arith::IndexCastUIOp");
    auto indexCast = dyn_cast<arith::IndexCastUIOp>(arg.getDefiningOp());
    auto nonIndexArg = indexCast.getOperand();
    // Non index arg must be a signless integer, but it has to have been created
    // by converting a signed integer to signless
    assert(isa<hwarith::CastOp>(nonIndexArg.getDefiningOp()));
    auto signlessCast = cast<hwarith::CastOp>(nonIndexArg.getDefiningOp());
    auto hwarithValue = signlessCast.getOperand();
    auto cmpType = dyn_cast<IntegerType>(hwarithValue.getType());
    assert(cmpType);
    auto caseAttr = IntegerAttr::get(cmpType, caseVal);
    auto constant =
        hwarith::ConstantOp::create(rewriter, loc, cmpType, caseAttr);
    auto compareOp = hwarith::ICmpOp::create(
        rewriter, loc, hwarith::ICmpPredicate::eq, hwarithValue, constant);
    auto resultOp =
        scf::IfOp::create(rewriter, loc, op.getResultTypes(), compareOp, true);
    Block *thenBlock = &resultOp.getThenRegion().front();
    rewriter.inlineBlockBefore(&op.getCaseBlock(caseIdx), thenBlock,
                               thenBlock->begin());

    Block *elseBlock = &resultOp.getElseRegion().front();
    if (caseIdx == op.getNumCases() - 1) {
      rewriter.inlineBlockBefore(&op.getDefaultBlock(), elseBlock,
                                 elseBlock->begin());
    } else {
      OpBuilder::InsertionGuard guard{rewriter};
      rewriter.setInsertionPointToStart(elseBlock);
      scf::IfOp elseIf = convertCases(rewriter, op, caseIdx + 1);
      scf::YieldOp::create(rewriter, loc, elseIf.getResults());
    }
    return resultOp;
  }

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    scf::IfOp resOp = convertCases(rewriter, op, 0);
    rewriter.replaceOp(op, resOp);
    return success();
  }
};
} // anonymous namespace

namespace {
struct CoreDSLSwitchToIf
    : public mlir::shortnail::impl::CoreDSLSwitchToIfBase<CoreDSLSwitchToIf> {
  using CoreDSLSwitchToIfBase::CoreDSLSwitchToIfBase;
  // This needs to be a class rather than a lambda because we cannot have
  // templated lambdas and this needs to be explicitly instantiated with
  // multiple op types for the walk method
  template <typename OpType>
  struct RegionToSCFConverter {
    // So we can use Pass::getChildAnalysis
    friend struct CoreDSLSwitchToIf;
    CoreDSLSwitchToIf &pass;
    bool &changed;

    WalkResult operator()(OpType op) {
      ControlFlowToSCFTransformation transformation;
      DominanceInfo &dominanceInfo = pass.getChildAnalysis<DominanceInfo>(op);
      FailureOr<bool> res =
          transformCFGToSCF(op.getRegion(), transformation, dominanceInfo);
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
    // NOTE: Treenail currently always generates an scf.execute_region operation
    // around every cf.switch. Because of this, it is technically only necessary
    // to run this on scf::ExecuteRegionOp and func::FuncOp
    auto res = isaxOp->walk<ORDER>(
        RegionToSCFConverter<coredsl::InstructionOp>{*this, changed});
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    res = isaxOp->walk<ORDER>(
        RegionToSCFConverter<coredsl::AlwaysOp>{*this, changed});
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    res = isaxOp->walk<ORDER>(
        RegionToSCFConverter<coredsl::SpawnOp>{*this, changed});
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    res =
        isaxOp->walk<ORDER>(RegionToSCFConverter<func::FuncOp>{*this, changed});
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    res = isaxOp->walk<ORDER>(
        RegionToSCFConverter<scf::ExecuteRegionOp>{*this, changed});
    if (res == WalkResult::interrupt()) {
      return signalPassFailure();
    }
    if (!changed) {
      markAllAnalysesPreserved();
      // If nothing changed, there is no need to convert scf.index_switch to
      // scf.if, as we expect there to not be any index switch ops other than
      // the one generated by converting cf.switch ops
      return;
    }
    MLIRContext &ctx = getContext();
    // First convert the cf operations
    OpPassManager cfToSCFPM{isaxOp.getOperationName()};
    if (failed(runPipeline(cfToSCFPM, isaxOp))) {
      return signalPassFailure();
    }
    RewritePatternSet patterns{&ctx};
    ConversionTarget target{ctx};
    target.addLegalDialect<coredsl::CoreDSLDialect, arith::ArithDialect,
                           hw::HWDialect, hwarith::HWArithDialect,
                           scf::SCFDialect, func::FuncDialect>();
    target.addIllegalOp<scf::IndexSwitchOp>();
    patterns.insert<IndexSwitchToSCFIf>(&ctx);
    if (failed(applyFullConversion(isaxOp, target, std::move(patterns)))) {
      return signalPassFailure();
    }
    // Run a dead value removal pass, as the index casts are now dead
    // NOTE: workaround to prevent crashes when deleting the index cast in
    // IndexSwitchToSCFIf::run
    OpPassManager finalPass{isaxOp.getOperationName()};
    finalPass.addPass(createRemoveDeadValuesPass());
    if (failed(runPipeline(finalPass, isaxOp))) {
      return signalPassFailure();
    }
  }
};
} // anonymous namespace
