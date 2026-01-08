//===- CoreDSLDirectives.h - custom directives for the CoreDSL dialect ----===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#ifndef SHORTNAIL_DIALECT_COREDSL_COREDSLDIRECTIVES_H
#define SHORTNAIL_DIALECT_COREDSL_COREDSLDIRECTIVES_H

#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

namespace mlir::coredsl {

ParseResult
parseRangedAccess(OpAsmParser &parser, bool optional,
                  std::optional<OpAsmParser::UnresolvedOperand> &base,
                  Type &baseType, IntegerAttr &from, IntegerAttr &to);

void printRangedAccess(OpAsmPrinter &p, Operation *, bool optional, Value base,
                       Type baseType, IntegerAttr from, IntegerAttr to);

ParseResult parseIdxRange(OpAsmParser &parser, IntegerAttr &from,
                          IntegerAttr &to);

void printIdxRange(OpAsmPrinter &p, Operation *, IntegerAttr from,
                   IntegerAttr to);

template <class Result, class... Types>
ParseResult parseFancyFunctionalType(OpAsmParser &parser, Result &result,
                                     Types &...inputs);

template <class OP, class Result, class... Types>
void printFancyFunctionalType(OpAsmPrinter &p, OP &op, Result result,
                              Types... inputs);

} // namespace mlir::coredsl

// include the template definitions
#include "CoreDSLDirectives.tpp"

#endif
