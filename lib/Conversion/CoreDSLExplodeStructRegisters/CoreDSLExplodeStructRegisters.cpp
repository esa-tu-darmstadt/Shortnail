#include "mlir/Transforms/DialectConversion.h"
#include "shortnail/Conversion/Passes.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HWArith/HWArithOps.h"

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

// Traverses a struct typed register, calling callbacks during traversal
// The callbacks are called with a string that consists of the register name,
// concatenated with the member names, so we can either explode the struct,
// or reference the already exploded registers
template <typename ScalarValueAction, typename StructMemberEntryAction,
          typename StructMemberExitAction>
void explodeRegs(std::string &regName, hw::StructType type,
                 ScalarValueAction scalarValueAction,
                 StructMemberEntryAction structMemberEntryAction,
                 StructMemberExitAction structMemberExitAction) {
  const size_t regNameSize = regName.size();
  for (hw::StructType::FieldInfo fieldInfo : type.getElements()) {
    regName += "_";
    regName += fieldInfo.name.getValue();
    if (auto structType = llvm::dyn_cast<hw::StructType>(fieldInfo.type)) {
      structMemberEntryAction(structType, fieldInfo.name);
      explodeRegs(regName, structType, scalarValueAction,
                  structMemberEntryAction, structMemberExitAction);
      structMemberExitAction(structType, fieldInfo.name);
    } else {
      scalarValueAction(regName, fieldInfo.name,
                        llvm::cast<IntegerType>(fieldInfo.type));
    }
    // Reset the string for the next iteration
    regName.resize(regNameSize);
  }
}

static constexpr auto emptyStructMemberEntryExitAction = [](hw::StructType,
                                                            StringAttr) {};

template <typename ScalarValueAction,
          typename StructMemberEntryAction =
              decltype(emptyStructMemberEntryExitAction),
          typename StructMemberExitAction =
              decltype(emptyStructMemberEntryExitAction)>
void explodeRegs(StringRef regName, hw::StructType type,
                 ScalarValueAction scalarValueAction,
                 StructMemberEntryAction structMemberEntryAction =
                     emptyStructMemberEntryExitAction,
                 StructMemberExitAction structMemberExitAction =
                     emptyStructMemberEntryExitAction) {
  auto nameString = std::string(regName);
  return explodeRegs(nameString, type, scalarValueAction,
                     structMemberEntryAction, structMemberExitAction);
}

struct StructExploderPattern : public OpConversionPattern<coredsl::RegisterOp> {
  llvm::StringMap<hw::StructType> &symNameToType;
  llvm::StringMap<unsigned> &symNameToMaxIndexWidth;

  StructExploderPattern(MLIRContext *ctx,
                        llvm::StringMap<hw::StructType> &structTypes,
                        llvm::StringMap<unsigned> &symNameToMaxIndexWidth)
      : OpConversionPattern<coredsl::RegisterOp>(ctx),
        symNameToType{structTypes},
        symNameToMaxIndexWidth{symNameToMaxIndexWidth} {}

  LogicalResult
  matchAndRewrite(coredsl::RegisterOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto structType = llvm::dyn_cast<hw::StructType>(op.getRegType())) {
      auto numElements = op.getNumElementsAttr();
      StringRef name = op.getName();
      rewriter.setInsertionPointAfter(op);
      Location loc = op.getLoc();
      explodeRegs(
          name, structType,
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
      symNameToType.insert(std::make_pair(op.getSymName(), structType));
      symNameToMaxIndexWidth.insert(
          std::make_pair(op.getSymName(), op.getMaxIndexWidth()));
      op.getMaxIndexWidth();
      rewriter.eraseOp(op);
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

struct StructRewriteSetOps : public OpConversionPattern<coredsl::SetOp> {
  const llvm::StringMap<hw::StructType> &symNameToType;
  const llvm::StringMap<unsigned> &symNameToMaxIndexWidth;

  StructRewriteSetOps(MLIRContext *ctx,
                      llvm::StringMap<hw::StructType> &structTypes,
                      llvm::StringMap<unsigned> &symNameToMaxIndexWidth)
      : OpConversionPattern<coredsl::SetOp>(ctx), symNameToType{structTypes},
        symNameToMaxIndexWidth{symNameToMaxIndexWidth} {}

  LogicalResult
  matchAndRewrite(coredsl::SetOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getContext();
    auto base = op.getBase();
    auto from = op.getFromAttr();
    auto to = op.getToAttr();
    auto loc = op.getLoc();
    // Check if the symbol is one of the removed ones
    StringRef symbolName = op.getSym();
    auto found = symNameToType.find(symbolName);
    if (found != symNameToType.end()) {
      auto structType = found->second;
      if (to != nullptr) {
        // Handle ranged access: Because the input value is an integer of size
        // range-size * struct-size, we need to extract the relevant values
        // manually and assign them to the right scalar register
        auto value = op.getValue();
        auto idxType = IndexType::get(ctx);
        size_t currBitPos = 0;
        for (int64_t i = from.getInt(); i <= to.getInt(); ++i) {
          const IntegerAttr idxAttr =
              i == 0 ? IntegerAttr::get(
                           IntegerType::get(ctx, 1, IntegerType::Unsigned), 0)
                     : IntegerAttr::get(ctx, APSInt::get(i));
          auto offsetConstant = hwarith::ConstantOp::create(
              rewriter, loc, idxAttr.getType(), idxAttr);
          // TODO: type is probably wrong
          auto offsetIdx =
              hwarith::AddOp::create(rewriter, loc, {base, offsetConstant});
          const unsigned maxIndexWidth =
              symNameToMaxIndexWidth.find(symbolName)->second;
          auto regIdxType = IntegerType::get(
              ctx, std::min(offsetIdx.getType().getWidth(), maxIndexWidth),
              IntegerType::Unsigned);
          auto idxVal =
              hwarith::CastOp::create(rewriter, loc, regIdxType, offsetIdx);
          explodeRegs(
              symbolName, structType,
              [&rewriter, &currBitPos, &loc, &value, &idxVal, idxType,
               ctx](StringRef newRegName, StringAttr fieldName,
                    IntegerType type) {
                const size_t bitsBegin = currBitPos;
                const size_t bitsEnd = currBitPos + type.getWidth() - 1;
                const auto bitsBeginAttr = IntegerAttr::get(idxType, bitsBegin);
                const auto bitsEndAttr = IntegerAttr::get(idxType, bitsEnd);
                assert(!type.isSignless());
                IntegerType bitExtractResType =
                    type.isSigned() ? IntegerType::get(ctx, type.getWidth(),
                                                       IntegerType::Unsigned)
                                    : type;
                auto extractedBits = coredsl::BitExtractOp::create(
                    rewriter, loc, bitExtractResType, nullptr, bitsBeginAttr,
                    bitsEndAttr, value);
                Operation *valueToWrite = extractedBits;
                if (bitExtractResType != type) {
                  valueToWrite = coredsl::CastOp::create(rewriter, loc, type,
                                                         extractedBits);
                }
                coredsl::SetOp::create(rewriter, loc, idxVal, nullptr, nullptr,
                                       newRegName, valueToWrite->getResult(0));
                currBitPos += type.getWidth();
              });
        }
      } else {
        SmallVector<Operation *> opStack{op.getValue().getDefiningOp()};
        explodeRegs(
            symbolName, structType,
            [&rewriter, &opStack, &loc, &base, &from, &to](
                StringRef newRegName, StringAttr fieldName, IntegerType type) {
              auto writtenValue = opStack.back();
              auto extractOp = hw::StructExtractOp::create(
                  rewriter, loc, writtenValue->getResult(0), fieldName);
              coredsl::SetOp::create(rewriter, loc, base, from, to, newRegName,
                                     extractOp->getResult(0));
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
      }
      rewriter.eraseOp(op);
      return LogicalResult::success();
    }
    return LogicalResult::failure();
  }
};

struct StructRewriteGetOps : public OpConversionPattern<coredsl::GetOp> {
  const llvm::StringMap<hw::StructType> &symNameToType;
  const llvm::StringMap<unsigned> &symNameToMaxIndexWidth;

  StructRewriteGetOps(MLIRContext *ctx,
                      llvm::StringMap<hw::StructType> &structTypes,
                      llvm::StringMap<unsigned> &symNameToMaxIndexWidth)
      : OpConversionPattern<coredsl::GetOp>(ctx), symNameToType{structTypes},
        symNameToMaxIndexWidth{symNameToMaxIndexWidth} {}

  LogicalResult
  matchAndRewrite(coredsl::GetOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = getContext();
    auto type = op.getResult().getType();
    // Check if sym is one of the exploded structs
    auto found = symNameToType.find(op.getSym());
    if (found != symNameToType.end()) {
      auto structType = found->second;
      StringRef symbolName = op.getSym();

      auto base = op.getBase();
      auto from = op.getFromAttr();
      auto to = op.getToAttr();
      auto loc = op.getLoc();
      Value replacement = nullptr;
      if (to != nullptr) {
        // Handle ranged access: Because the return value is a scalar value in
        // this case, read all scalar values from the exploded registers and
        // concatenate them using comb.concat
        assert(from);
        SmallVector<Value> toConcatenate;
        const unsigned maxIndexWidth =
            symNameToMaxIndexWidth.find(symbolName)->second;
        for (int64_t i = from.getInt(); i <= to.getInt(); ++i) {
          APInt val{64, (uint64_t)i, true};
          val = val.trunc(std::max(val.getActiveBits(), 1u));
          auto offsetType =
              IntegerType::get(ctx, val.getBitWidth(), IntegerType::Signed);
          auto offset = hwarith::ConstantOp::create(
              rewriter, loc, offsetType, IntegerAttr::get(offsetType, val));
          // result needs to be unsigned and respect access size
          auto addRes = hwarith::AddOp::create(rewriter, loc, {base, offset});
          auto idxType = IntegerType::get(
              ctx, std::min(addRes.getType().getWidth(), maxIndexWidth),
              IntegerType::Unsigned);
          auto newBase =
              hwarith::CastOp::create(rewriter, loc, idxType, addRes);
          // TODO: are the values in the right order?
          explodeRegs(
              symbolName, structType,
              [&rewriter, &loc, &newBase, &toConcatenate,
               ctx](StringRef newRegName, StringAttr fieldName,
                    IntegerType type) {
                auto gotValue = coredsl::GetOp::create(
                    rewriter, loc, type, newBase, nullptr, nullptr, newRegName);
                auto gotType = cast<IntegerType>(gotValue.getType());
                Operation *result = gotValue;
                if (gotType.getSignedness() != IntegerType::Signless) {
                  auto signlessType = IntegerType::get(ctx, gotType.getWidth(),
                                                       IntegerType::Signless);
                  result = hwarith::CastOp::create(rewriter, loc, signlessType,
                                                   gotValue);
                }
                toConcatenate.push_back(result->getResult(0));
              });
        }
        auto result = comb::ConcatOp::create(rewriter, loc, toConcatenate);
        IntegerType resultSignlessType = cast<IntegerType>(result.getType());
        auto resultCast = hwarith::CastOp::create(
            rewriter, loc,
            IntegerType::get(ctx, resultSignlessType.getWidth(),
                             IntegerType::Unsigned),
            result);
        replacement = resultCast.getResult();
      } else {
        SmallVector<hw::StructCreateOp> structOps;
        SmallVector<Value> structMembers;
        SmallVector<size_t> structBeginIndices = {0};
        explodeRegs(
            symbolName, structType,
            [&rewriter, &loc, &structMembers, &base, &from, &to](
                StringRef newRegName, StringAttr fieldName, IntegerType type) {
              auto gotValue = coredsl::GetOp::create(rewriter, loc, type, base,
                                                     from, to, newRegName);
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
        replacement = finalStruct.getResult();
      }
      rewriter.replaceOp(op, replacement);
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
    llvm::StringMap<hw::StructType> symToTypeMap;
    llvm::StringMap<unsigned> symToMaxIndexWidthMap;
    patterns.insert<StructExploderPattern>(&ctx, symToTypeMap,
                                           symToMaxIndexWidthMap);
    ConversionTarget target{ctx};
    target.addLegalDialect<hwarith::HWArithDialect, comb::CombDialect,
                           hw::HWDialect, coredsl::CoreDSLDialect>();
    target.addDynamicallyLegalOp<coredsl::RegisterOp>(
        [](coredsl::RegisterOp op) { return op.getElementType().isInteger(); });
    if (failed(applyPartialConversion(isax, target, std::move(patterns)))) {
      return signalPassFailure();
    }
    patterns.clear();
    target.addDynamicallyLegalOp<coredsl::GetOp>(
        [&symToTypeMap](coredsl::GetOp op) {
          return symToTypeMap.find(op.getSym()) == symToTypeMap.end();
        });
    target.addDynamicallyLegalOp<coredsl::SetOp>(
        [&symToTypeMap](coredsl::SetOp op) {
          return symToTypeMap.find(op.getSym()) == symToTypeMap.end();
        });
    patterns.insert<StructRewriteGetOps, StructRewriteSetOps>(
        &ctx, symToTypeMap, symToMaxIndexWidthMap);

    if (failed(applyPartialConversion(isax, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
