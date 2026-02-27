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
    Merges multiple `coredsl.isax` operations within a module into a single
    unified `coredsl.isax` named "merged".

    Shared architectural state (core_x registers, core_pc, core_fp, core_mem
    address spaces) is deduplicated — one canonical declaration is kept and all
    references are remapped. The pass asserts that shared state elements have
    matching types and sizes across ISAXes.

    ISAX-private state (local registers, aliases, helper functions) is prefixed
    with "MERGED{N}" (where N is a per-ISAX index) to avoid name collisions,
    then moved into the merged ISAX. Instructions and always-blocks are moved
    directly.

    If only a single `coredsl.isax` is present, the pass is a no-op.
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
    Transpiles a `coredsl.isax` operation into a Python module suitable for use
    as a cocotb simulation model. Each instruction is emitted as a Python
    function that mirrors the CoreDSL behavior using bitwise operations on
    Python integers.

    CoreDSL operations (`coredsl.get`, `coredsl.set`, `coredsl.cast`,
    `coredsl.bitextract`, `coredsl.concat`, bitwise/shift/arithmetic ops) and
    `hwarith` operations are translated to their Python equivalents with
    explicit bit-width masking. Control flow (`scf.if`, `scf.for`) is lowered
    to Python `if`/`for` statements.

    Register and address space accesses are emitted as method calls on a
    simulation context object, enabling the generated Python to interface with
    the cocotb testbench infrastructure.
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
