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

// IDEA: make pattern be applied recursively?!?!
// - Would generate more registers that are later deleted
// - may be easier

// TODO: not sure if twine makes sense here, because we are appending here
template <typename ScalarValueAction, typename StructMemberAction>
void explodeRegs(StringRef regName, hw::StructType type,
                 ConversionPatternRewriter &rewriter,
                 ScalarValueAction scalarValueAction,
                 StructMemberAction structMemberAction) {
  for (hw::StructType::FieldInfo fieldInfo : type.getElements()) {
    //auto newRegName = regName + "_" + fieldInfo.name.getValue();
    auto newRegName = std::string(regName);
    newRegName += "_";
    newRegName += fieldInfo.name.getValue();
    if (auto structType = llvm::dyn_cast<hw::StructType>(fieldInfo.type)) {
      structMemberAction(structType, fieldInfo.name);
      explodeRegs(newRegName, structType, rewriter, scalarValueAction,
                  structMemberAction);
    } else {
      scalarValueAction(newRegName, fieldInfo.name,
                        llvm::cast<IntegerType>(fieldInfo.type));
    }
  }
}

struct StructExploderPattern : public OpConversionPattern<coredsl::RegisterOp> {
  // TODO: may not be necessary
  llvm::StringMap<coredsl::RegisterOp> &nameToRegMap;
  StructExploderPattern(MLIRContext *ctx,
                        llvm::StringMap<coredsl::RegisterOp> &nameToRegMap)
      : OpConversionPattern<coredsl::RegisterOp>(ctx),
        nameToRegMap{nameToRegMap} {}

  LogicalResult
  matchAndRewrite(coredsl::RegisterOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto structType = llvm::dyn_cast<hw::StructType>(op.getRegType())) {
      StringRef name = op.getName();
      rewriter.setInsertionPointAfter(op);
      Location loc = op.getLoc();
      nameToRegMap.insert(std::make_pair(name, op));
      explodeRegs(
          name, structType, rewriter,
          [this, &rewriter, &loc, &op](StringRef newRegName, StringAttr fieldName,
                                  IntegerType fieldType) {
            auto ctx = rewriter.getContext();
            StringAttr symbolName = StringAttr::get(ctx, newRegName);
            auto reg = coredsl::RegisterOp::create(
                rewriter, loc, {}, symbolName, op.getIsConst(), op.getIsVolatile(),
                /*numElements=*/nullptr, {},
                fieldType,
                op.getAccessMode());

            nameToRegMap.insert(std::make_pair(newRegName, reg));
          },
          [](hw::StructType, StringAttr) {});
      // TODO: not sure if this will work, as the reg is still used
      rewriter.eraseOp(op);
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

struct StructRewriteSetOps : public OpConversionPattern<coredsl::SetOp> {
  const llvm::StringMap<coredsl::RegisterOp> &nameToRegMap;

  StructRewriteSetOps(MLIRContext *ctx,
                      const llvm::StringMap<coredsl::RegisterOp> &nameToRegMap)
      : OpConversionPattern<coredsl::SetOp>(ctx), nameToRegMap{nameToRegMap} {}

  LogicalResult
  matchAndRewrite(coredsl::SetOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    if (auto structType = llvm::dyn_cast<hw::StructType>(value.getType())) {
      StringRef symbolName = op.getSym();
      auto loc = op.getLoc();
      SmallVector<Operation *> opStack{op.getValue().getDefiningOp()};
      explodeRegs(
          symbolName, structType, rewriter,
          [&rewriter, &opStack, &loc](StringRef newRegName,
                                      StringAttr fieldName, IntegerType type) {
            auto writtenValue = opStack.back();
            auto op = coredsl::SetOp::create(rewriter, loc, nullptr, nullptr, nullptr,
                                   newRegName, writtenValue->getResult(0));
            llvm::outs() << "New op: " << op << "\n";
            // TODO: this is writing struct inject
            llvm::outs() << "Set val: " << *writtenValue << "\n";
          },
          [&rewriter, &opStack, &loc](hw::StructType type,
                                      StringAttr fieldName) {
            // TODO: emit hw.struct_extract and push result on stack
            // TODO: get extracted value
            auto toExtractFrom = opStack.back();
            Value structVal = toExtractFrom->getResult(0);
            assert(llvm::isa<hw::StructType>(structVal.getType()));
            auto extractOp = hw::StructExtractOp::create(
                rewriter, loc, toExtractFrom->getResult(0), fieldName);
            opStack.push_back(extractOp);
          });
      rewriter.eraseOp(op);
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

struct StructRewriteGetOps : public OpConversionPattern<coredsl::GetOp> {
  const llvm::StringMap<coredsl::RegisterOp> &nameToRegMap;

  StructRewriteGetOps(MLIRContext *ctx,
                      const llvm::StringMap<coredsl::RegisterOp> &nameToRegMap)
      : OpConversionPattern<coredsl::GetOp>(ctx), nameToRegMap{nameToRegMap} {}

  LogicalResult
  matchAndRewrite(coredsl::GetOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op.getResult().getType();
    if (auto structType = llvm::dyn_cast<hw::StructType>(type)) {
      StringRef symbolName = op.getSym();
      coredsl::RegisterOp accessedReg = nameToRegMap.find(symbolName)->second;

      auto loc = op.getLoc();
      SmallVector<Value> structMembers;
      // TODO: need to combine the gotten vavlues into a struct
      explodeRegs(
          symbolName, structType, rewriter,
          [&rewriter, &loc, &structMembers](StringRef newRegName, StringAttr fieldName, IntegerType type) {
            auto gotValue = coredsl::GetOp::create(rewriter, loc, type, nullptr, nullptr, nullptr, newRegName);
            structMembers.push_back(gotValue.getResult());
          },
          [&rewriter, &loc, &structMembers](hw::StructType type, StringAttr fieldName) {
            // TODO: hope this does not scramble struct members
            auto structVal = hw::StructCreateOp::create(rewriter, loc, type, structMembers);
            structMembers.clear();
            structMembers.push_back(structVal.getResult());
          });
      // TODO: hope this does not scramble struct members
      auto finalStruct = hw::StructCreateOp::create(rewriter, loc, accessedReg.getElementType(), structMembers);
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
    patterns.insert<StructExploderPattern>(&ctx, nameToRegMap);
    ConversionTarget target{ctx};
    target.addLegalDialect<hw::HWDialect, coredsl::CoreDSLDialect>();
    target.addDynamicallyLegalOp<coredsl::RegisterOp>([](coredsl::RegisterOp op){
      return op.getElementType().isInteger();
    });
    if (failed(applyPartialConversion(isax, target, std::move(patterns)))) {
      return signalPassFailure();
    }
    patterns.clear();
    target.addDynamicallyLegalOp<coredsl::GetOp>([](coredsl::GetOp op){
      return op.getResult().getType().isInteger();
    });
    target.addDynamicallyLegalOp<coredsl::SetOp>([](coredsl::SetOp op){
      return op.getValue().getType().isInteger();
    });
    patterns.insert<StructRewriteGetOps, StructRewriteSetOps>(&ctx,
                                                              nameToRegMap);
    if (failed(applyPartialConversion(isax, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
