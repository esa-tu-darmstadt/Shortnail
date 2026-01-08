//===- CoreDSLDirectives.cpp - CoreDSL custom directive implementations ---===//
//
// Copyright 2022 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#include "shortnail/Dialect/CoreDSL/CoreDSLDirectives.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::coredsl {

//===----------------------------------------------------------------------===//
// custom<IdxRange>
//===----------------------------------------------------------------------===//

ParseResult parseIdxRange(OpAsmParser &parser, IntegerAttr &from,
                          IntegerAttr &to) {
  // The idx range access is completely optional
  if (parser.parseOptionalLSquare()) {
    return success();
  }

  if (parser.parseAttribute(from, parser.getBuilder().getIndexType())) {
    return failure();
  }

  if (succeeded(parser.parseOptionalColon()) &&
      parser.parseAttribute(to, parser.getBuilder().getIndexType())) {
    return failure();
  }

  if (parser.parseRSquare()) {
    return failure();
  }

  return success();
}

void printIdxRange(OpAsmPrinter &p, Operation *, IntegerAttr from,
                   IntegerAttr to) {
  if (from) {
    p << '[';
    p << from.getValue();
    if (to) {
      p << ':';
      p << to.getValue();
    }
    p << ']';
  }
}

//===----------------------------------------------------------------------===//
// custom<RangedAccess>
//===----------------------------------------------------------------------===//

ParseResult
parseRangedAccess(OpAsmParser &parser, bool optional,
                  std::optional<OpAsmParser::UnresolvedOperand> &base,
                  Type &baseType, IntegerAttr &from, IntegerAttr &to) {
  // `[` ($base^ `:` type($base) `,`)? ($from^ (`:` $to^)?)? `]`
  if (optional) {
    // The ranged access is completely optional
    if (parser.parseOptionalLSquare()) {
      return success();
    }
  } else {
    if (parser.parseLSquare()) {
      return failure();
    }
  }

  bool hasBase;
  OpAsmParser::UnresolvedOperand baseOperand;
  auto baseParseRes = parser.parseOptionalOperand(baseOperand);
  hasBase = baseParseRes.has_value();
  if (hasBase) {
    if (failed(*baseParseRes)) {
      return failure();
    }
    if (parser.parseColonType(baseType)) {
      return failure();
    }
    base = baseOperand;
  }

  auto hasSep = succeeded(parser.parseOptionalComma());
  auto loc = parser.getCurrentLocation();

  bool hasRange;
  auto fromRangeRes =
      parser.parseOptionalAttribute(from, parser.getBuilder().getIndexType());
  hasRange = fromRangeRes.has_value();
  if (hasRange) {
    if (failed(*fromRangeRes)) {
      return failure();
    }

    if (hasBase != hasSep) {
      return parser.emitError(loc, hasSep ? "unexpected seperator `,`"
                                          : "expected `,` as seperator");
    }

    if (succeeded(parser.parseOptionalColon())) {
      if (parser.parseAttribute(to, parser.getBuilder().getIndexType())) {
        return failure();
      }
    }
  }

  loc = parser.getCurrentLocation();
  if (parser.parseRSquare()) {
    return failure();
  }

  if (!hasBase && !hasRange) {
    return parser.emitError(loc, "an empty ranged access is not allowed");
  }

  return success();
}

void printRangedAccess(OpAsmPrinter &p, Operation *, bool optional, Value base,
                       Type baseType, IntegerAttr from, IntegerAttr to) {
  bool hasBase = static_cast<bool>(base);
  bool hasRangeAccess = static_cast<bool>(from);

  if (hasBase | hasRangeAccess) {
    p << '[';
    if (hasBase) {
      p.printOperand(base);
      p << ' ' << ':' << ' ';
      p.printType(baseType);
    }
    if (hasRangeAccess) {
      if (hasBase) {
        p << ',' << ' ';
      }
      p << from.getValue();
      if (to) {
        p << ':';
        p << to.getValue();
      }
    }
    p << ']';
  }
}

} // namespace mlir::coredsl
