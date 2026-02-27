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

def AnalyzeISAX : Pass<"analyze-isax", "mlir::ModuleOp"> {
  let summary = "Analyze CoreDSL ISAX MLIR and output structured YAML";
  let description = [{
    Analyzes CoreDSL MLIR (ShortNail dialect) and outputs a structured YAML
    description of each ISAX instruction's encoding, operand semantics, and
    side effects. Expects exactly one coredsl.ISAXOp in the module.
    The YAML is consumed by the Python patch generator (gen_patches.py).
  }];
  let constructor = "::mlir::shortnail::createAnalyzeISAXPass()";
  let dependentDialects = [
    "::mlir::coredsl::CoreDSLDialect",
    "::circt::hwarith::HWArithDialect",
    "::mlir::func::FuncDialect"
  ];
  let options = [
    Option<"outputPath", "output", "std::string", /*default=*/"\"\"",
           "Output path for the YAML analysis file">
  ];
}

#endif // SHORTNAIL_CONVERSION_PASSES_TD
