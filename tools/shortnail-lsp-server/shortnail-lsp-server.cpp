//===- shortnail-lsp-server.cpp --------------------------------------------===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#include "circt/InitAllDialects.h"
#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  circt::registerAllDialects(registry);
  registry.insert<coredsl::CoreDSLDialect>();
  return failed(MlirLspServerMain(argc, argv, registry));
}
