//===- shortnail-opt.cpp ---------------------------------------------------===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#include "shortnail/Conversion/Passes.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.h"

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<coredsl::CoreDSLDialect>();

  mlir::func::registerInlinerExtension(registry);
  mlir::registerCSEPass();
  mlir::registerSCCPPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();

  mlir::shortnail::registerConversionPasses();

  return failed(
      mlir::MlirOptMain(argc, argv, "Shortnail optimizer driver\n", registry));
}
