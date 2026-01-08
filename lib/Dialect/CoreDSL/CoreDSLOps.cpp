//===- CoreDSLOps.cpp - CoreDSL dialect ops -------------------------------===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLDirectives.h"

#include "circt/Dialect/HWArith/HWArithOps.h"
#include "circt/Dialect/HWArith/HWArithTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/APSInt.h"

using namespace circt::hwarith;

namespace mlir {
namespace coredsl {

template <typename SymbolTy>
static SymbolTy lookupSymbolInModule(Operation *op, StringRef name) {
  // look for the next (parent) operation in hierarchy with a symbol table;
  // for the CoreDSL dialect, this will be the enclosing the MLIR module.
  auto isaxOp = op->template getParentOfType<coredsl::ISAXOp>();
  assert(isaxOp);
  return dyn_cast_or_null<SymbolTy>(SymbolTable::lookupSymbolIn(isaxOp, name));
}

//===----------------------------------------------------------------------===//
// InstructionOp
//===----------------------------------------------------------------------===//

static bool isValidEncodingLiteralChar(char c) { return c == '0' || c == '1'; }

bool isEncodingFieldLiteral(StringRef field) {
  return isValidEncodingLiteralChar(field[0]);
}

StringRef InstructionOp::getInstructionArgumentName(size_t argNo) {
  for (auto enc : getEncoding().getValue()) {
    auto fieldName = cast<StringAttr>(enc).getValue();
    if (fieldName.empty()) {
      continue;
    }

    if (!isEncodingFieldLiteral(fieldName)) {
      if (argNo == 0) {
        // Remove the '%' from the argument name otherwise it will be printed as
        // hexadecimal!
        return fieldName.drop_front();
      }
      --argNo;
    }
  }

  return StringRef();
}

void InstructionOp::getAsmBlockArgumentNames(Region &region,
                                             OpAsmSetValueNameFn setNameFn) {
  auto &block = region.front();
  for (size_t i = 0, e = block.getNumArguments(); i != e; ++i) {
    auto name = getInstructionArgumentName(i);
    if (name.empty()) {
      llvm_unreachable("Invalid instruction argument number!");
    } else {
      setNameFn(block.getArgument(i), name);
    }
  }
}

void InstructionOp::getEncodingFields(SmallVectorImpl<EncodingField> &fields) {
  fields.clear();
  ArrayRef<Attribute> enc = getEncoding().getValue();
  unsigned bitIdx = 0;
  unsigned argIdx = getNumArguments() - 1;
  for (auto revIt = enc.rbegin(), revEnd = enc.rend(); revIt != revEnd;
       ++revIt) {
    StringAttr fieldAttr = cast<StringAttr>(*revIt);
    StringRef field = fieldAttr.getValue();

    if (field.empty())
      continue;

    if (isEncodingFieldLiteral(field)) {
      bitIdx += field.size();
    } else {
      unsigned argSize = getArgument(argIdx).getType().getIntOrFloatBitWidth();
      fields.push_back({bitIdx, bitIdx + argSize - 1});
      bitIdx += argSize;
      --argIdx;
    }
  }
  // TODO We could also rewrite the algorithm above, but this requires knowledge
  //      of the instruction word size
  std::reverse(fields.begin(), fields.end());
}

StringAttr InstructionOp::getEncodingMask() {
  std::string s;

  auto argTys = getFunctionType().getInputs();
  ArrayRef<Attribute> encodingFields = getEncoding().getValue();

  unsigned opIdx = 0;
  for (auto e : encodingFields) {
    StringAttr fieldAttr = cast<StringAttr>(e);
    StringRef field = fieldAttr.getValue();

    if (field.empty())
      continue;

    if (isEncodingFieldLiteral(field)) {
      s += field.str();
    } else {
      s += std::string(argTys[opIdx].getIntOrFloatBitWidth(), '-');
      ++opIdx;
    }
  }

  return StringAttr::get(getContext(), s);
}

static ParseResult
parseISAEncoding(OpAsmParser &parser,
                 SmallVectorImpl<OpAsmParser::Argument> &entryArgs,
                 SmallVectorImpl<Attribute> &encodingAttrs,
                 SmallVectorImpl<Type> &opTypesOnly) {
  auto parseArgument = [&]() -> ParseResult {
    llvm::SMLoc loc = parser.getCurrentLocation();

    // Parse argument name if present.
    OpAsmParser::Argument argument;
    auto parseRes = parser.parseOptionalArgument(argument, /* allowType=*/true);
    if (parseRes.has_value() && succeeded(parseRes.value()) &&
        !argument.ssaName.name.empty()) {

      entryArgs.push_back(argument);
      opTypesOnly.push_back(argument.type);

      // This name contains the '%'
      auto regName =
          StringAttr::get(parser.getContext(), argument.ssaName.name);
      encodingAttrs.push_back(regName);
    } else {
      StringAttr encodingField;
      OptionalParseResult parseRes =
          parser.parseOptionalAttribute(encodingField);

      // Try parsing an encoding field!
      if (parseRes.has_value() && succeeded(*parseRes)) {

        // Verify that the encoding string only consists of valid encoding
        // chars!
        for (auto digit : encodingField) {
          if (!isValidEncodingLiteralChar(digit)) {
            return parser.emitError(
                loc,
                "invalid character in encoding, only '0' and '1' are valid");
          }
        }

        encodingAttrs.push_back(encodingField);
      } else {
        return parser.emitError(loc,
                                "expected SSA identifier or encoding string");
      }
    }

    return success();
  };

  // Parse the function arguments.
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::Paren,
                                     parseArgument))
    return failure();

  return success();
}

ParseResult InstructionOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr name;
  SmallVector<OpAsmParser::Argument, 8> entryArgs;
  SmallVector<Attribute, 8> encodingAttrs;
  SmallVector<Type, 8> opTypesOnly;

  if (failed(parser.parseSymbolName(name, SymbolTable::getSymbolAttrName(),
                                    result.attributes)))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (failed(parseISAEncoding(parser, entryArgs, encodingAttrs, opTypesOnly)))
    return failure();

  auto &builder = parser.getBuilder();
  auto funcTy = builder.getFunctionType(opTypesOnly, ArrayRef<Type>());
  result.addAttribute(InstructionOp::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(funcTy));

  result.addAttribute(InstructionOp::getEncodingAttrName(result.name),
                      ArrayAttr::get(result.getContext(), encodingAttrs));

  auto *behavior = result.addRegion();
  if (parser.parseRegion(*behavior, entryArgs))
    return failure();

  return success();
}

void InstructionOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getName());

  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{
                              SymbolTable::getSymbolAttrName(),
                              InstructionOp::getFunctionTypeAttrName(),
                              InstructionOp::getEncodingAttrName()});

  auto args = getArguments();
  auto argTys = getFunctionType().getInputs();
  p << '(';

  ArrayRef<Attribute> encodingFields = getEncoding().getValue();

  unsigned opIdx = 0;
  bool first = true;
  for (auto e : encodingFields) {
    StringAttr fieldAttr = cast<StringAttr>(e);
    StringRef field = fieldAttr.getValue();

    if (field.empty())
      continue;

    if (!first) {
      p << ", ";
    }

    if (isEncodingFieldLiteral(field)) {
      p << '"' << field << '"';
    } else {
      p.printOperand(args[opIdx]);
      p << " : ";
      p.printType(argTys[opIdx]);
      ++opIdx;
    }

    first = false;
  }

  p << ')';
  p.printRegion(getFunctionBody(), false);
}

LogicalResult InstructionOp::verify() {
  unsigned encodingBits = 0;
  int argIndex = getNumArguments() - 1;
  for (auto field : getEncoding().getAsValueRange<StringAttr>()) {
    if (field.empty())
      continue;

    if (isEncodingFieldLiteral(field)) {
      encodingBits += field.size();
    } else if (field[0] == '%') {
      if (argIndex < 0) {
        return emitError(
            "encoding contains more arguments than the instruction");
      }

      auto type = getArgument(argIndex).getType();

      if (!isHWArithIntegerType(type)) {
        return emitError(
            "encoding argument type must be an arbitrary precision "
            "integer with signedness semantics");
      }

      encodingBits += type.getIntOrFloatBitWidth();
      --argIndex;
    } else {
      // TODO this case shouldn't be reachable
      return emitError("Invalid encoding field: ") << field.str();
    }
  }

  if (argIndex >= 0) {
    return emitError("encoding contains fewer arguments than the instruction");
  }
  if (encodingBits != 32) {
    return emitError("encoding is not 32-bit wide"); // TODO don't hardcode RV32
  }
  return success();
}

LogicalResult InstructionOp::verifyType() {
  auto type = getFunctionTypeAttr().getValue();
  if (!isa<FunctionType>(type))
    return emitOpError("requires '" + getFunctionTypeAttrName().getValue() +
                       "' attribute of function type");
  return success();
}

//===----------------------------------------------------------------------===//
// AddressSpaceOp
//===----------------------------------------------------------------------===//

LogicalResult AddressSpaceOp::verify() {
  if (getAccessMode() != AddressSpaceAccessMode::wire) {
    if (!getAddrType().has_value()) {
      return emitError("this protocol expects an address type");
    }
  } else {
    if (getAddrType().has_value()) {
      return emitError("the wire protocol has no address type");
    }
  }

  if (!isHWArithIntegerType(getResType())) {
    return emitError("the result type must be an arbitrary precision integer "
                     "with signedness semantics");
  }
  if (getAddrType().has_value() &&
      !isHWArithIntegerType(getAddrType().value())) {
    return emitError("the address type must be an arbitrary precision integer "
                     "with signedness semantics");
  }

  unsigned enforceAddrWidth = 0;
  unsigned enforceResWidth = 0;
  switch (getAccessMode()) {
  case AddressSpaceAccessMode::wire: {
    // restrict core_mem to wire to () -> ui1
    // TODO hardcoded values
    enforceResWidth = 1;
    break;
  }
  case AddressSpaceAccessMode::core_mem: {
    // restrict core_mem to (ui32) -> ui8
    // TODO hardcoded values
    enforceAddrWidth = 32;
    enforceResWidth = 8;
    break;
  }
  case AddressSpaceAccessMode::core_csr: {
    // restrict core_csr to (ui12) -> ui32
    // TODO hardcoded values
    enforceAddrWidth = 12;
    enforceResWidth = 32;
    break;
  }
  default:
    break;
  }

  if (enforceResWidth != 0 &&
      getResType().getIntOrFloatBitWidth() != enforceResWidth) {
    return emitError("the result type is restricted to ")
           << enforceResWidth << " bits";
  }
  if (enforceAddrWidth != 0 &&
      getAddrType()->getIntOrFloatBitWidth() != enforceAddrWidth) {
    return emitError("the address type is restricted to ")
           << enforceAddrWidth << " bits";
  }

  return success();
}

uint64_t AddressSpaceOp::getMemSize() {
  // We do not know the exact memory size
  return 0;
}
unsigned AddressSpaceOp::getMaxIndexWidth() {
  return getAddrType() ? getAddrType()->getIntOrFloatBitWidth() : 0;
}
unsigned AddressSpaceOp::getMinIndexWidth() {
  return 0;
  // TODO restrict to the address width?
  // return getAddrType()->getIntOrFloatBitWidth();
}
IntegerType AddressSpaceOp::getElementType() {
  return cast<IntegerType>(getResType());
}

//===----------------------------------------------------------------------===//
// RegisterOp
//===----------------------------------------------------------------------===//

bool RegisterOp::isShimed() {
  return getAccessMode() != RegisterAccessMode::local;
}
bool RegisterOp::isRegField() {
  return getNumElements() && getNumElements()->getZExtValue() > 1;
}
uint64_t RegisterOp::getSize() {
  return isRegField() ? getNumElements()->getZExtValue() : 1;
}

LogicalResult RegisterOp::fold(FoldAdaptor adaptor,
                               SmallVectorImpl<OpFoldResult> &results) {
  if (getNumElements() && getSize() == 1) {
    // Replace the "register field" of size one with an normal register!
    removeNumElementsAttr();
    return success();
  }

  return failure();
}

static unsigned getAPIntBitWidth(const APInt &value, bool sign) {
  if (sign) {
    return value.getSignificantBits();
  }
  return value.getActiveBits();
}

uint64_t RegisterOp::getMemSize() { return getSize(); }
unsigned RegisterOp::getMaxIndexWidth() {
  return llvm::Log2_64_Ceil(getSize());
}
unsigned RegisterOp::getMinIndexWidth() { return 0; }
IntegerType RegisterOp::getElementType() {
  return cast<IntegerType>(getRegType());
}

LogicalResult RegisterOp::verify() {
  // Regfield checks
  if (getNumElements() && getNumElements()->getZExtValue() == 0) {
    return emitError("register fields of size 0 are invalid");
  }

  if (!isHWArithIntegerType(getRegType())) {
    return emitError("register type must be an arbitrary precision integer "
                     "with signedness semantics");
  }

  // Initializer checks
  if (ArrayAttr initializer = getInitializerAttr()) {
    auto initValues = initializer.getAsValueRange<IntegerAttr>();
    unsigned initSize = std::distance(initValues.begin(), initValues.end());
    if (initSize != getSize())
      return emitError(
          "number of elements in initializer does not match register size");

    // check that the values do not exceed the type size
    unsigned regTypeWidth = getRegType().getIntOrFloatBitWidth();
    for (auto iv : initValues) {
      unsigned ivWidth =
          getAPIntBitWidth(iv, cast<IntegerType>(getRegType()).isSigned());
      if (ivWidth > regTypeWidth) {
        return emitError("initial value width exceeds register width: ")
               << iv.getSExtValue() << " (" << ivWidth << " bits)";
      }
    }
  } else {
    if (getIsConst())
      return emitError("Const registers must be initialized");
  }

  // Keyword sanity checks
  if (isRegField() && getAccessMode() == RegisterAccessMode::core_pc) {
    return emitError(
        "the program counter (core_pc) keyword may only be used for "
        "single registers");
  }

  if (!isRegField() && getAccessMode() == RegisterAccessMode::core_x) {
    return emitError(
        "the default register file must be declared as register field");
  }

  if (!isRegField() && getAccessMode() == RegisterAccessMode::core_fp) {
    return emitError(
        "the floating point register file must be declared as register field");
  }

  return success();
}

static ParseResult parseInitializer(OpAsmParser &parser, ArrayAttr &attr) {
  auto &builder = parser.getBuilder();
  if (failed(parser.parseOptionalEqual()))
    // No initializer!
    return success();

  SmallVector<int64_t> values;
  auto parseInt = [&]() -> ParseResult {
    int64_t v;
    auto res = parser.parseOptionalInteger(v);
    if (!res.has_value() || failed(*res))
      return failure();
    values.push_back(v);
    return success();
  };

  if (succeeded(parseInt()) || succeeded(parser.parseCommaSeparatedList(
                                   AsmParser::Delimiter::Square, parseInt))) {
    attr = builder.getIndexArrayAttr(values);
    return success();
  }

  return failure();
}

static void printInitializer(OpAsmPrinter &p, Operation *op, ArrayAttr attr) {
  if (!attr)
    return;

  p << " = ";
  auto values = attr.getValue();

  if (values.size() == 1) {
    p << cast<IntegerAttr>(values.front()).getAPSInt().getSExtValue();
    return;
  }

  p << "[";
  llvm::interleaveComma(values, p, [&](Attribute v) {
    p << cast<IntegerAttr>(v).getAPSInt().getSExtValue();
  });
  p << "]";
}

//===----------------------------------------------------------------------===//
// BitSetOp, BitExtractOp
//===----------------------------------------------------------------------===//

template <class BitOpTy>
static LogicalResult checkBitAccessOperation(BitOpTy op) {
  if (!op.getFrom() && !op.getBase()) {
    return op->emitError(
        "at least a base index or bit index range must be specified");
  }

  auto valueWidth = op.getValue().getType().getIntOrFloatBitWidth();
  auto accessWidth = op.getAccessWidth();

  if (accessWidth > valueWidth) {
    return op->emitError("bit index range exceeds input value width");
  }

  if (auto base = op.getBase()) {
    //  given the accessWidth we can further restrict the allowed index width
    auto maxIdxWidth = llvm::Log2_64_Ceil(valueWidth - accessWidth + 1);
    if (base.getType().getIntOrFloatBitWidth() > maxIdxWidth) {
      return op->emitError(
                 "base address width exceeds the max required index width of ")
             << maxIdxWidth;
    }
  }

  if (auto from = op.getFrom()) {
    if (from->getZExtValue() >= valueWidth) {
      return op->emitError("bit index 'from' exceeds input value width");
    }

    if (auto to = op.getTo()) {
      if (to->getZExtValue() >= valueWidth) {
        return op->emitError("bit index 'to' exceeds input value width");
      }
    }
  }

  return success();
}

LogicalResult BitExtractOp::verify() {
  if (failed(checkBitAccessOperation(*this))) {
    return failure();
  }

  auto accessWidth = getAccessWidth();
  auto resultType = getResult().getType();

  // check the result types
  if (!resultType.isUnsignedInteger(accessWidth)) {
    return emitError(
        "the result type for a read access must be unsigned and of the "
        "same size as the index range");
  }

  return success();
}

LogicalResult BitExtractOp::canonicalize(BitExtractOp op,
                                         PatternRewriter &rewriter) {

  // If the min idx is zero then we have a truncation that can be folded
  // to a CastOp!
  // However, this transformation is only valid if getFrom() >= getTo() holds.
  // Otherwise, the bit order is expected to be in reversed order!
  if (!op.getBase()) {
    if (auto to = op.getTo()) {
      if (to->getZExtValue() == 0) {
        rewriter.replaceOpWithNewOp<CastOp>(op, op.getType(), op.getValue());
        return success();
      }
    }
  }

  return failure();
}

LogicalResult BitSetOp::verify() {
  if (failed(checkBitAccessOperation(*this))) {
    return failure();
  }

  auto accessWidth = getAccessWidth();
  auto valueType = getValue().getType();
  auto resultType = getResult().getType();

  // check the rhs operand and result types
  if (getRhs().getType().getIntOrFloatBitWidth() != accessWidth) {
    return emitError("bit index range does not equal the write operand width");
  }

  // For a write access the result type should be identical to the lhs
  if (valueType != resultType) {
    return emitError(
        "the result type for a write access must be the same as the "
        "value type");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attrs, mlir::OpaqueProperties properties,
    mlir::RegionRange regions, SmallVectorImpl<Type> &results) {

  const auto lhs = cast<IntegerType>(operands[0].getType());
  const auto rhs = cast<IntegerType>(operands[1].getType());

  // Bit width rules are taken from:
  // https://github.com/Minres/CoreDSL/wiki/Expressions#concatenation
  const unsigned operandsTotalWidth = rhs.getWidth() + lhs.getWidth();

  // the result of a concat op is always unsigned!
  results.push_back(
      IntegerType::get(context, operandsTotalWidth, IntegerType::Unsigned));
  return success();
}

//===----------------------------------------------------------------------===//
// GetOp, SetOp
//===----------------------------------------------------------------------===//

template <typename AccessOpTy>
static LogicalResult checkAccess(AccessOpTy op, Type requiredType) {
  if (auto info = op.getMemInfo()) {
    // Calculate the expected type: element type width * access width while
    // keeping the signedness
    Type expectedType = IntegerType::get(
        op.getContext(), info->elementType.getWidth() * op.getAccessWidth(),
        info->elementType.getSignedness());
    if (expectedType != requiredType) {
      return op.emitError("type mismatch, referencing storage of type ")
             << info->elementType << " with access width of "
             << op.getAccessWidth();
    }

    bool isIndexable = info->maxIdxWidth > 0;

    if ((op.getBase() || op.getFrom()) != isIndexable) {
      if (isIndexable)
        return op.emitError("missing index variable or range to access this "
                            "register/address space");
      else
        return op.emitError("index and ranged accesses are not allowed for "
                            "this register/address space");
    }

    // check that neither the offsets nor the access width violate the max index
    // width!
    if (auto from = op.getFrom()) {
      auto reqBitWidth = getAPIntBitWidth(from.value(), false);
      if (reqBitWidth > info->maxIdxWidth)
        return op.emitError("`from` offset exceeds the register/address "
                            "space's address width of ")
               << info->maxIdxWidth << " with a required offset bit width of "
               << reqBitWidth;

      uint64_t startOffset = from.value().getZExtValue();
      if (auto to = op.getTo()) {
        auto reqBitWidth = getAPIntBitWidth(to.value(), false);
        if (reqBitWidth > info->maxIdxWidth)
          return op.emitError("`to` offset exceeds the register/address "
                              "space's address width of ")
                 << info->maxIdxWidth << " with a required offset bit width of "
                 << reqBitWidth;

        if (llvm::Log2_64_Ceil(op.getAccessWidth()) > info->maxIdxWidth)
          return op.emitError("the access width exceeds the address width of "
                              "register/address space");

        startOffset = std::min(to.value().getZExtValue(), startOffset);
      }

      if (info->size != 0 && startOffset + op.getAccessWidth() > info->size) {
        return op.emitError("the access range exceeds the bounds of the "
                            "register/address space");
      }
    }

    //  access for register field
    if (auto accessIdx = op.getBase()) {
      auto accessIdxWidth = accessIdx.getType().getIntOrFloatBitWidth();
      if (accessIdxWidth > info->maxIdxWidth)
        return op.emitError(
            "index width exceeds address width of register/address space");

      if (accessIdxWidth < info->minIdxWidth)
        return op.emitError("index width does not meet the min address width "
                            "to access the register/address space");
    }
  } else {
    return op.emitError(
        "must reference a 'coredsl.register' or 'coredsl.addrspace'");
  }

  return success();
}

static Operation *genericResolveSymbol(Operation *op, StringRef sym) {
  if (auto alias = lookupSymbolInModule<AliasOp>(op, sym)) {
    return alias;
  }
  return lookupSymbolInModule<GetSetOpInterface>(op, sym);
}

static RegisterOp fullyResolveReference(Operation *op, unsigned &startOffset) {
  if (auto regOp = dyn_cast<RegisterOp>(op))
    return regOp;
  if (isa<AliasOp>(op)) {
    auto alias = cast<AliasOp>(op);
    startOffset += alias.getRangeStart();
    return fullyResolveReference(genericResolveSymbol(alias, alias.getRef()),
                                 startOffset);
  }
  return nullptr;
}

LogicalResult GetOp::canonicalize(GetOp op, PatternRewriter &rewriter) {
  // Perform constant propagation, replace by a constant if feasible!
  unsigned initIdx = 0;
  if (auto accessIdx = op.getBase()) {
    auto *defOp = accessIdx.getDefiningOp();
    if (!defOp || !defOp->hasTrait<OpTrait::ConstantLike>())
      return failure();

    assert(isa<ConstantOp>(defOp));
    initIdx = cast<ConstantOp>(defOp).getRawValue().getZExtValue();
  }

  // For now we only handle single element constant accesses
  if (!op.hasSingleIdxAccess())
    return failure();

  if (auto from = op.getFrom())
    initIdx += from->getZExtValue();

  auto resolvedSym = op.resolveSymbol();
  resolvedSym = fullyResolveReference(resolvedSym, initIdx);
  if (!resolvedSym)
    return failure();

  // Non const target -> exit!
  assert(isa<GetSetOpInterface>(resolvedSym));
  assert(isa<RegisterOp>(resolvedSym));
  if (!cast<RegisterOp>(resolvedSym).getIsConst())
    return failure();

  auto regOp = cast<RegisterOp>(resolvedSym);
  auto initOpt = regOp.getInitializer();
  assert(initOpt.has_value());
  auto constVal = cast<IntegerAttr>(initOpt->getValue()[initIdx]);
  auto resType = op->getResult(0).getType();
  rewriter.replaceOpWithNewOp<ConstantOp>(
      op, resType,
      rewriter.getIntegerAttr(resType, constVal.getValue().getSExtValue()));

  return success();
}

LogicalResult GetOp::verify() {
  return checkAccess(*this, getResult().getType());
}

LogicalResult SetOp::verify() {
  if (auto info = getMemInfo()) {
    if (info->isConst) {
      return emitError("writing to a const 'coredsl.register', "
                       "'coredsl.addrspace' or 'coredsl.alias' is prohibited");
    }
  }

  return checkAccess(*this, getValue().getType());
}

static std::optional<MemInfo> genericGetMemInfo(Operation *op, StringRef sym) {
  if (auto alias = lookupSymbolInModule<AliasOp>(op, sym)) {
    return alias.getMemInfo();
  }
  if (auto mem = lookupSymbolInModule<GetSetOpInterface>(op, sym)) {
    MemInfo info;
    info.size = mem.getMemSize();
    info.isConst = mem.isConst();
    info.isVolatile = mem.isVolatile();
    info.maxIdxWidth = mem.getMaxIndexWidth();
    info.minIdxWidth = mem.getMinIndexWidth();
    info.elementType = mem.getElementType();
    return info;
  }
  return std::nullopt;
}

std::optional<MemInfo> GetOp::getMemInfo() {
  return genericGetMemInfo(*this, getSym());
}
Operation *GetOp::resolveSymbol() {
  return genericResolveSymbol(*this, getSym());
}
std::optional<MemInfo> SetOp::getMemInfo() {
  return genericGetMemInfo(*this, getSym());
}
Operation *SetOp::resolveSymbol() {
  return genericResolveSymbol(*this, getSym());
}

//===----------------------------------------------------------------------===//
// AliasOp
//===----------------------------------------------------------------------===//

std::optional<MemInfo> AliasOp::getMemInfo() {
  MemInfo info;
  info.isConst = getIsConst();
  info.isVolatile = getIsVolatile();
  auto aliasRange = getAliasRange();
  info.size = aliasRange;
  info.maxIdxWidth = aliasRange ? llvm::Log2_64_Ceil(aliasRange) : 0;
  info.minIdxWidth = 0;
  info.elementType = resolveSymbol().getElementType();
  return info;
}

GetSetOpInterface AliasOp::resolveSymbol() {
  return lookupSymbolInModule<GetSetOpInterface>(*this, getRef());
}

LogicalResult AliasOp::verify() {
  if (auto from = getFrom()) {
    if (auto to = getTo()) {
      if (to->ugt(from.value())) {
        return emitError(
            "The `to` index may not be larger than the `from` index!");
      }
    }
  }

  if (auto resolvedSym = resolveSymbol()) {
    auto maxIdxBitWidth = resolvedSym.getMaxIndexWidth();
    bool isIndexable = maxIdxBitWidth > 0;
    auto aliasRange = getAliasRange();

    if (isIndexable != (aliasRange > 0)) {
      if (isIndexable) {
        return emitError("missing alias range for this 'coredsl.register' or "
                         "'coredsl.addrspace'");
      } else {
        return emitError("no alias range may be used for this "
                         "'coredsl.register' or 'coredsl.addrspace'");
      }
    }

    auto memSize = resolvedSym.getMemSize();

    if (auto from = getFrom()) {
      auto reqBitWidth = getAPIntBitWidth(from.value(), false);
      if (maxIdxBitWidth < reqBitWidth) {
        return emitError(
            "the end index of the specified alias range exceeds the "
            "'coredsl.register' or 'coredsl.addrspace' index width.");
      }
    }

    auto rangeStart = getRangeStart();
    if (memSize != 0 && rangeStart + aliasRange > memSize) {
      return emitError("the specified alias range exceeds the underlying size "
                       "of the 'coredsl.register' or 'coredsl.addrspace'");
    }
  } else {
    return emitError(
        "must reference a 'coredsl.register' or 'coredsl.addrspace'");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// BitOp
//===----------------------------------------------------------------------===//

static IntegerType::SignednessSemantics
getSignedInheritedSignedness(IntegerType lhs, IntegerType rhs) {
  // Signed operands are dominant and enforce a signed result
  if (lhs.getSignedness() == rhs.getSignedness()) {
    // the signedness is also identical to the operands
    return lhs.getSignedness();
  } else {
    // For mixed signedness the result is always signed
    return IntegerType::Signed;
  }
}

static LogicalResult inferBitOpTypes(MLIRContext *context,
                                     std::optional<Location> loc,
                                     ValueRange operands, DictionaryAttr attrs,
                                     mlir::OpaqueProperties properties,
                                     mlir::RegionRange regions,
                                     SmallVectorImpl<Type> &results) {
  auto lhs = cast<IntegerType>(operands[0].getType());
  auto rhs = cast<IntegerType>(operands[1].getType());

  // Bit width rules are taken from:
  // https://github.com/Minres/CoreDSL/wiki/Expressions#bitwise-and-or-xor

  // The result width is always the max operand bit width
  const unsigned resultWidth = std::max(lhs.getWidth(), rhs.getWidth());

  const IntegerType::SignednessSemantics signedness =
      getSignedInheritedSignedness(lhs, rhs);

  results.push_back(IntegerType::get(context, resultWidth, signedness));

  return success();
}

//===----------------------------------------------------------------------===//
// ModOp
//===----------------------------------------------------------------------===//

LogicalResult ModOp::inferReturnTypes(MLIRContext *context,
                                      std::optional<Location> loc,
                                      ValueRange operands, DictionaryAttr attrs,
                                      mlir::OpaqueProperties properties,
                                      mlir::RegionRange regions,
                                      SmallVectorImpl<Type> &results) {

  auto lhs = cast<IntegerType>(operands[0].getType());
  auto rhs = cast<IntegerType>(operands[1].getType());

  // Bit width rules are taken from:
  // https://github.com/Minres/CoreDSL/wiki/Expressions#modulusremainder

  // The signedness of lhs is identical to the result!
  const IntegerType::SignednessSemantics signedness = lhs.getSignedness();

  unsigned resultWidth;
  if (lhs.getSignedness() == rhs.getSignedness()) {
    // the signedness is also identical to the operands
    resultWidth = std::min(lhs.getWidth(), rhs.getWidth());
  } else {
    // For mixed signedness the result bit width is a bit more complicated
    if (lhs.isSigned()) {
      resultWidth = std::min(lhs.getWidth(), rhs.getWidth() + 1);
    } else {
      resultWidth = std::max(1u, std::min(lhs.getWidth(), rhs.getWidth() - 1));
    }
  }

  results.push_back(IntegerType::get(context, resultWidth, signedness));

  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {

  auto resultType = getResult().getType();

  // Ensure that only single bit values may be casted to a signless
  // representation
  if (!isHWArithIntegerType(resultType) &&
      (!resultType.isSignlessInteger(1) ||
       getValue().getType().getIntOrFloatBitWidth() != 1)) {
    return emitError(
        "casting to a signless integer is only allowed from SI1/UI1 to I1");
  }

  return success();
}

} // namespace coredsl
} // namespace mlir

#define GET_OP_CLASSES
#include "shortnail/Dialect/CoreDSL/CoreDSL.cpp.inc"
