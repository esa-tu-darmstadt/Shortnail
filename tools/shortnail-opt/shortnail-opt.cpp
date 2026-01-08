//===- shortnail-opt.cpp ---------------------------------------------------===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#include "shortnail/Conversion/Passes.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.h"

#include "circt/InitAllDialects.h"
#include "circt/InitAllPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace circt;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<coredsl::CoreDSLDialect>();
  registry.insert<comb::CombDialect, hw::HWDialect, seq::SeqDialect,
                  hwarith::HWArithDialect, sv::SVDialect>();
  registry.insert<arith::ArithDialect, memref::MemRefDialect, func::FuncDialect,
                  scf::SCFDialect>();

  mlir::func::registerInlinerExtension(registry);
  mlir::registerCSE();
  mlir::registerSCCP();
  mlir::registerInliner();
  mlir::registerCanonicalizer();

  mlir::shortnail::registerConversionPasses();

  return failed(
      mlir::MlirOptMain(argc, argv, "Shortnail optimizer driver\n", registry));
}
