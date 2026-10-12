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

// The callbacks are called with a string that consists of the register name,
// concatenated with the member names, so we can either explode the register,
// or reference the already exploded registers by name
template <typename ScalarValueAction, typename StructMemberEntryAction,
          typename StructMemberExitAction>
void traverseStructReg(std::string &regName, hw::StructType type,
                       ScalarValueAction scalarValueAction,
                       StructMemberEntryAction structMemberEntryAction,
                       StructMemberExitAction structMemberExitAction) {
  const size_t regNameSize = regName.size();
  for (hw::StructType::FieldInfo fieldInfo : type.getElements()) {
    regName += "_";
    regName += fieldInfo.name.getValue();
    if (auto structType = llvm::dyn_cast<hw::StructType>(fieldInfo.type)) {
      structMemberEntryAction(structType, fieldInfo.name);
      traverseStructReg(regName, structType, scalarValueAction,
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
void traverseStructReg(StringRef regName, hw::StructType type,
                       ScalarValueAction scalarValueAction,
                       StructMemberEntryAction structMemberEntryAction =
                           emptyStructMemberEntryExitAction,
                       StructMemberExitAction structMemberExitAction =
                           emptyStructMemberEntryExitAction) {
  auto nameString = std::string(regName);
  return traverseStructReg(nameString, type, scalarValueAction,
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
      traverseStructReg(
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
          });
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

static IntegerType getMinIntegerType(MLIRContext *ctx, int64_t value) {
  auto signedness = IntegerType::Unsigned;
  // llvm::bit_width is for unsigned integers only, so check how many bits
  // are necessary for the positive value and then add one for the sign bit
  if (value < 0) {
    value = -value;
    signedness = IntegerType::Signed;
  }
  int bitWidth = llvm::bit_width(static_cast<uint64_t>(value)) +
                 (signedness == IntegerType::Signed);
  bitWidth = std::max(1, bitWidth);
  return IntegerType::get(ctx, bitWidth, signedness);
}

// Emits add of base to offset, truncating to maxIndexWidth if the result type
// is larger
static Value emitTruncatedOffset(ConversionPatternRewriter &rewriter,
                                 MLIRContext *ctx, Value base, int64_t offset,
                                 unsigned maxIndexWidth, Location loc) {
  if (offset == 0 && base != nullptr) {
    // We don't need to insert a cast here, because if base is larger than
    // maxIndexWidth, the original op was not valid in the first place
    return base;
  }
  const auto offsetType = getMinIntegerType(ctx, offset);
  const auto offsetAttr = IntegerAttr::get(offsetType, offset);
  auto offsetConstant = hwarith::ConstantOp::create(
      rewriter, loc, offsetAttr.getType(), offsetAttr);
  Operation *index = offsetConstant;
  if (base != nullptr) {
    index = hwarith::AddOp::create(rewriter, loc, {base, offsetConstant});
  }
  auto indexType = index->getResult(0).getType();
  const unsigned indexWidth = indexType.getIntOrFloatBitWidth();
  const unsigned neededWidth = std::min(indexWidth, maxIndexWidth);
  if (indexWidth > neededWidth || indexType.isSignedInteger()) {
    auto idxType = IntegerType::get(ctx, neededWidth, IntegerType::Unsigned);
    return hwarith::CastOp::create(rewriter, loc, idxType, index->getResult(0));
  }
  return index->getResult(0);
}

static Value concatenateValues(ConversionPatternRewriter &rewriter,
                               Location loc, const SmallVector<Value> &vals) {
  Value currentValue = vals[0];
  for (size_t i = 1, end = vals.size(); i < end; ++i) {
    currentValue =
        coredsl::ConcatOp::create(rewriter, loc, currentValue, vals[i]);
  }
  return currentValue;
}

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
        const size_t numElements = std::abs(to.getInt() - from.getInt()) + 1;
        size_t currBitPos = numElements * *structType.getBitWidth();
        const unsigned maxIndexWidth =
            symNameToMaxIndexWidth.find(symbolName)->second;
        const int64_t step = from.getInt() <= to.getInt() ? 1 : -1;
        const int64_t end = to.getInt() + step;
        for (int64_t i = from.getInt(); i != end; i += step) {
          auto idxVal =
              emitTruncatedOffset(rewriter, ctx, base, i, maxIndexWidth, loc);
          traverseStructReg(
              symbolName, structType,
              [&rewriter, &currBitPos, &loc, &value, &idxVal, idxType,
               ctx](StringRef newRegName, StringAttr fieldName,
                    IntegerType type) {
                currBitPos -= type.getWidth();
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
                    rewriter, loc, bitExtractResType, nullptr, bitsEndAttr,
                    bitsBeginAttr, value);
                Value valueToWrite = extractedBits->getResult(0);
                if (bitExtractResType != type) {
                  valueToWrite = coredsl::CastOp::create(rewriter, loc, type,
                                                         valueToWrite);
                }
                coredsl::SetOp::create(rewriter, loc, idxVal, nullptr, nullptr,
                                       newRegName, valueToWrite);
              });
        }
      } else {
        // Single element access: Extract the struct members and set the
        // exploded scalar registers
        SmallVector<Value> valueStack{op.getValue()};
        traverseStructReg(
            symbolName, structType,
            [&rewriter, &valueStack, &loc, &base, &from, &to](
                StringRef newRegName, StringAttr fieldName, IntegerType type) {
              auto writtenValue = valueStack.back();
              auto extractOp = hw::StructExtractOp::create(
                  rewriter, loc, writtenValue, fieldName);
              coredsl::SetOp::create(rewriter, loc, base, from, to, newRegName,
                                     extractOp->getResult(0));
            },
            [&rewriter, &valueStack, &loc](hw::StructType type,
                                           StringAttr fieldName) {
              auto toExtractFrom = valueStack.back();
              Value structVal = toExtractFrom;
              assert(llvm::isa<hw::StructType>(structVal.getType()));
              auto extractOp = hw::StructExtractOp::create(
                  rewriter, loc, toExtractFrom, fieldName);
              valueStack.push_back(extractOp);
            },
            [&valueStack](hw::StructType, StringAttr) {
              valueStack.pop_back();
            });
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
        // Ranged access: Because the return value is a scalar value in this
        // case, read all scalar values from the exploded registers and
        // concatenate them using comb.concat
        assert(from);
        SmallVector<Value> toConcatenate;
        const unsigned maxIndexWidth =
            symNameToMaxIndexWidth.find(symbolName)->second;
        const int64_t step = from.getInt() <= to.getInt() ? 1 : -1;
        const int64_t end = to.getInt() + step;
        for (int64_t i = from.getInt(); i != end; i += step) {
          auto newBase =
              emitTruncatedOffset(rewriter, ctx, base, i, maxIndexWidth, loc);
          traverseStructReg(
              symbolName, structType,
              [&rewriter, &loc, &newBase, &toConcatenate](StringRef newRegName,
                                                          StringAttr fieldName,
                                                          IntegerType type) {
                auto gotValue = coredsl::GetOp::create(
                    rewriter, loc, type, newBase, nullptr, nullptr, newRegName);
                toConcatenate.push_back(gotValue);
              });
        }
        replacement = concatenateValues(rewriter, loc, toConcatenate);
      } else {
        // Single element access: Read the scalar values and bundle them into a
        // struct
        SmallVector<Value> structMembers;
        // Stores the indices in structMembers from which a nested struct's
        // members begin, so they can be bundled into an intermediate struct
        SmallVector<size_t> structBeginIndices = {};
        traverseStructReg(
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
    // Because we are deleting the register ops when exploding the registers,
    // we need to retain the struct type and the max index width for the passes
    // that handle the get / set ops
    llvm::StringMap<hw::StructType> symNameToType;
    llvm::StringMap<unsigned> symNameToMaxIndexWidth;
    patterns.insert<StructExploderPattern>(&ctx, symNameToType,
                                           symNameToMaxIndexWidth);
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
        [&symNameToType](coredsl::GetOp op) {
          return symNameToType.find(op.getSym()) == symNameToType.end();
        });
    target.addDynamicallyLegalOp<coredsl::SetOp>(
        [&symNameToType](coredsl::SetOp op) {
          return symNameToType.find(op.getSym()) == symNameToType.end();
        });
    patterns.insert<StructRewriteGetOps, StructRewriteSetOps>(
        &ctx, symNameToType, symNameToMaxIndexWidth);

    if (failed(applyPartialConversion(isax, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
