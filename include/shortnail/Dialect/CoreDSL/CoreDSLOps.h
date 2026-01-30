//===- CoreDSLOps.h - CoreDSL dialect ops -----------------------*- C++ -*-===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#ifndef SHORTNAIL_DIALECT_COREDSL_COREDSLOPS_H
#define SHORTNAIL_DIALECT_COREDSL_COREDSLOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLInterfaces.h.inc"

namespace mlir::coredsl {
struct MemInfo {
  uint64_t size;
  unsigned maxIdxWidth;
  unsigned minIdxWidth;
  IntegerType elementType;
  bool isConst;
  bool isVolatile;
};

bool isEncodingFieldLiteral(StringRef field);
} // namespace mlir::coredsl

#define GET_OP_CLASSES
#include "shortnail/Dialect/CoreDSL/CoreDSL.h.inc"

#endif // SHORTNAIL_DIALECT_COREDSL_COREDSLOPS_H
