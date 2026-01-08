//===- CoreDSLToPy.h - Translating CoreDSL to Python ------------*- C++ -*-===//
//
// Copyright 2025 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#ifndef SHORTNAIL_CONVERSION_COREDSLTOPY_H
#define SHORTNAIL_CONVERSION_COREDSLTOPY_H

#include <memory>

namespace mlir {

class Pass;
class Operation;

namespace shortnail {

void deleteUnusedSymbols(Operation *startOp);
std::unique_ptr<mlir::Pass> createCoreDSLToPyPass();

} // namespace shortnail
} // namespace mlir

#endif // SHORTNAIL_CONVERSION_COREDSLTOPY_H
