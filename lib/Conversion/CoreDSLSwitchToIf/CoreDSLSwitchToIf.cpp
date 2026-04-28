#include "mlir/Transforms/DialectConversion.h"
#include "shortnail/Conversion/Passes.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HWArith/HWArithDialect.h"
#include "circt/Dialect/HWArith/HWArithOps.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

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
    int64_t caseVal = op.getCases()[caseIdx];
    // Use arith for comparison, because hwarith::ICmp does not work with index
    // types
    arith::ConstantIndexOp constant =
        arith::ConstantIndexOp::create(rewriter, loc, caseVal);
    arith::CmpIOp compareOp = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, constant, op.getArg());
    scf::IfOp resultOp =
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

  void runOnOperation() override {
    auto isaxOp = getOperation();
    MLIRContext &ctx = getContext();
    // First convert the cf operations
    OpPassManager cfToSCFPM{isaxOp.getOperationName()};
    cfToSCFPM.addPass(createCoreDSLLiftCFToSCF());
    if (failed(runPipeline(cfToSCFPM, isaxOp))) {
      return signalPassFailure();
    }
    RewritePatternSet patterns{&ctx};
    ConversionTarget target{ctx};
    target.addLegalDialect<coredsl::CoreDSLDialect, arith::ArithDialect,
                           hw::HWDialect, hwarith::HWArithDialect,
                           scf::SCFDialect>();
    target.addIllegalOp<scf::IndexSwitchOp>();
    patterns.insert<IndexSwitchToSCFIf>(&ctx);
    if (failed(applyFullConversion(isaxOp, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // anonymous namespace
