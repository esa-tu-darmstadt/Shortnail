//===- CoreDSLDialect.cpp - CoreDSL dialect -------------------------------===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::coredsl;

struct CoreDSLInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //
  // Unconditionally allow inlining, for now.
  //===--------------------------------------------------------------------===//

  virtual bool isLegalToInline(Operation *call, Operation *callable,
                               bool wouldBeCloned) const override {
    return true;
  }

  virtual bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                               IRMapping &valueMapping) const override {
    return true;
  }

  virtual bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                               IRMapping &valueMapping) const override {
    return true;
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator(std.return) by replacing it with a new
  /// operation as necessary.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // Only "std.return" needs to be handled here.
    auto returnOp = cast<func::ReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// Attempts to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    return nullptr;
  }
};

//===----------------------------------------------------------------------===//
// CoreDSL dialect.
//===----------------------------------------------------------------------===//

void CoreDSLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "shortnail/Dialect/CoreDSL/CoreDSL.cpp.inc"
      >();
  addInterfaces<CoreDSLInlinerInterface>();
}

#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.cpp.inc"
#include "shortnail/Dialect/CoreDSL/CoreDSLEnums.cpp.inc"
#include "shortnail/Dialect/CoreDSL/CoreDSLInterfaces.cpp.inc"
