#include "mlir/Transforms/DialectConversion.h"
#include "shortnail/Conversion/Passes.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"

namespace mlir {
namespace shortnail {
#define GEN_PASS_DEF_COREDSLEXPLODESTRUCTREGISTERS
#include "shortnail/Conversion/Passes.h.inc"
} // namespace shortnail
} // namespace mlir

using namespace mlir;
using namespace mlir::shortnail;
using namespace circt;

namespace {

template <typename ScalarValueAction, typename StructMemberEntryAction,
          typename StructMemberExitAction>
void explodeRegs(StringRef regName, hw::StructType type,
                 ConversionPatternRewriter &rewriter,
                 ScalarValueAction scalarValueAction,
                 StructMemberEntryAction structMemberEntryAction,
                 StructMemberExitAction structMemberExitAction) {
  for (hw::StructType::FieldInfo fieldInfo : type.getElements()) {
    auto newRegName = std::string(regName);
    newRegName += "_";
    newRegName += fieldInfo.name.getValue();
    if (auto structType = llvm::dyn_cast<hw::StructType>(fieldInfo.type)) {
      structMemberEntryAction(structType, fieldInfo.name);
      explodeRegs(newRegName, structType, rewriter, scalarValueAction,
                  structMemberEntryAction, structMemberExitAction);
      structMemberExitAction(structType, fieldInfo.name);
    } else {
      scalarValueAction(newRegName, fieldInfo.name,
                        llvm::cast<IntegerType>(fieldInfo.type));
    }
  }
}

struct StructExploderPattern : public OpConversionPattern<coredsl::RegisterOp> {
  StructExploderPattern(MLIRContext *ctx)
      : OpConversionPattern<coredsl::RegisterOp>(ctx) {}

  LogicalResult
  matchAndRewrite(coredsl::RegisterOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto structType = llvm::dyn_cast<hw::StructType>(op.getRegType())) {
      auto numElements = op.getNumElementsAttr();
      StringRef name = op.getName();
      rewriter.setInsertionPointAfter(op);
      Location loc = op.getLoc();
      explodeRegs(
          name, structType, rewriter,
          [&rewriter, &loc, &op, &numElements](StringRef newRegName,
                                               StringAttr fieldName,
                                               IntegerType fieldType) {
            auto ctx = rewriter.getContext();
            StringAttr symbolName = StringAttr::get(ctx, newRegName);
            coredsl::RegisterOp::create(rewriter, loc, {}, symbolName,
                                        op.getIsConst(), op.getIsVolatile(),
                                        numElements, {}, fieldType,
                                        op.getAccessMode());
          },
          [](hw::StructType, StringAttr) {}, [](hw::StructType, StringAttr) {});
      rewriter.eraseOp(op);
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

struct StructRewriteSetOps : public OpConversionPattern<coredsl::SetOp> {
  StructRewriteSetOps(MLIRContext *ctx)
      : OpConversionPattern<coredsl::SetOp>(ctx) {}

  LogicalResult
  matchAndRewrite(coredsl::SetOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    auto base = op.getBase();
    auto from = op.getFromAttr();
    auto to = op.getToAttr();
    if (auto structType = llvm::dyn_cast<hw::StructType>(value.getType())) {
      StringRef symbolName = op.getSym();
      auto loc = op.getLoc();
      SmallVector<Operation *> opStack{op.getValue().getDefiningOp()};
      explodeRegs(
          symbolName, structType, rewriter,
          [&rewriter, &opStack, &loc, &base, &from, &to](StringRef newRegName,
                                      StringAttr fieldName, IntegerType type) {
            auto writtenValue = opStack.back();
            auto extractOp = hw::StructExtractOp::create(
                rewriter, loc, writtenValue->getResult(0), fieldName);
            coredsl::SetOp::create(rewriter, loc, base, from, to,
                                   newRegName, extractOp->getResult(0));
          },
          [&rewriter, &opStack, &loc](hw::StructType type,
                                      StringAttr fieldName) {
            auto toExtractFrom = opStack.back();
            Value structVal = toExtractFrom->getResult(0);
            assert(llvm::isa<hw::StructType>(structVal.getType()));
            auto extractOp = hw::StructExtractOp::create(
                rewriter, loc, toExtractFrom->getResult(0), fieldName);
            opStack.push_back(extractOp);
          },
          [&opStack](hw::StructType, StringAttr) { opStack.pop_back(); });
      rewriter.eraseOp(op);
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

struct StructRewriteGetOps : public OpConversionPattern<coredsl::GetOp> {
  StructRewriteGetOps(MLIRContext *ctx)
      : OpConversionPattern<coredsl::GetOp>(ctx) {}

  LogicalResult
  matchAndRewrite(coredsl::GetOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getResult().getType();
    if (auto structType = llvm::dyn_cast<hw::StructType>(type)) {
      StringRef symbolName = op.getSym();

      auto base = op.getBase();
      auto from = op.getFromAttr();
      auto to = op.getToAttr();
      auto loc = op.getLoc();
      SmallVector<Value> structMembers;
      SmallVector<size_t> structBeginIndices = {0};
      explodeRegs(
          symbolName, structType, rewriter,
          [&rewriter, &loc, &structMembers, &base, &from, &to](
              StringRef newRegName, StringAttr fieldName, IntegerType type) {
            auto gotValue = coredsl::GetOp::create(
                rewriter, loc, type, base, from, to, newRegName);
            structMembers.push_back(gotValue.getResult());
          },
          [&structBeginIndices, &structMembers](hw::StructType, StringAttr) {
            structBeginIndices.push_back(structMembers.size());
          },
          [&rewriter, &loc, &structBeginIndices,
           &structMembers](hw::StructType type, StringAttr fieldName) {
            const size_t structBeginIdx = structBeginIndices.back();
            auto currStructMembers = ArrayRef(
                structMembers.begin() + structBeginIdx, structMembers.end());
            auto structVal = hw::StructCreateOp::create(rewriter, loc, type,
                                                        currStructMembers);
            structMembers.resize(structBeginIdx);
            structBeginIndices.pop_back();
            structMembers.push_back(structVal.getResult());
          });
      auto finalStruct =
          hw::StructCreateOp::create(rewriter, loc, type, structMembers);
      rewriter.replaceOp(op, finalStruct.getResult());
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

struct CoreDSLExplodeStructRegisters
    : public mlir::shortnail::impl::CoreDSLExplodeStructRegistersBase<
          CoreDSLExplodeStructRegisters> {
  using CoreDSLExplodeStructRegistersBase::CoreDSLExplodeStructRegistersBase;

  void runOnOperation() override {
    coredsl::ISAXOp isax = getOperation();
    auto &ctx = getContext();
    RewritePatternSet patterns{&ctx};
    llvm::StringMap<coredsl::RegisterOp> nameToRegMap;
    llvm::StringMap<Type> nameToTypeMap;
    patterns.insert<StructExploderPattern>(&ctx);
    ConversionTarget target{ctx};
    target.addLegalDialect<hw::HWDialect, coredsl::CoreDSLDialect>();
    target.addDynamicallyLegalOp<coredsl::RegisterOp>(
        [](coredsl::RegisterOp op) { return op.getElementType().isInteger(); });
    if (failed(applyPartialConversion(isax, target, std::move(patterns)))) {
      return signalPassFailure();
    }
    patterns.clear();
    target.addDynamicallyLegalOp<coredsl::GetOp>(
        [](coredsl::GetOp op) { return op.getResult().getType().isInteger(); });
    target.addDynamicallyLegalOp<coredsl::SetOp>(
        [](coredsl::SetOp op) { return op.getValue().getType().isInteger(); });
    patterns.insert<StructRewriteGetOps, StructRewriteSetOps>(&ctx);

    if (failed(applyPartialConversion(isax, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
