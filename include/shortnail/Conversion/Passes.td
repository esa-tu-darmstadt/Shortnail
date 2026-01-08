//===-- Passes.td - Conversion pass definitions ------------*- tablegen -*-===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#ifndef SHORTNAIL_CONVERSION_PASSES_TD
#define SHORTNAIL_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def MergeIsax : Pass<"merge-multiple-isaxes", "mlir::ModuleOp"> {
  let summary = "Merge multiple MLIR CoreDSL files";
  let description = [{
    TODO
  }];
  let constructor = "::mlir::shortnail::createMergeISAXPass()";
  let dependentDialects = [
    "::mlir::coredsl::CoreDSLDialect",
    "::mlir::func::FuncDialect"
  ];
}

def CoreDSLToPy : Pass<"coredsl-to-python", "::mlir::coredsl::ISAXOp"> {
  let summary = "Transpile CoreDSL to Python.";
  let description = [{
    TODO
  }];
  let constructor = "::mlir::shortnail::createCoreDSLToPyPass()";
  let dependentDialects = [
    "::mlir::coredsl::CoreDSLDialect",
    "::circt::hw::HWDialect",
    "::circt::hwarith::HWArithDialect",
  ];
}

#endif // SHORTNAIL_CONVERSION_PASSES_TD
