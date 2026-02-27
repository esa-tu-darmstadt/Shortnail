//===- AnalyzeISAX.h - Analyze CoreDSL ISAX for LLVM patching ----*- C++ -*-==//
//
// Copyright 2026 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#ifndef SHORTNAIL_CONVERSION_ANALYZEISAX_H
#define SHORTNAIL_CONVERSION_ANALYZEISAX_H

#include <memory>

namespace mlir {

class Pass;

namespace shortnail {

std::unique_ptr<mlir::Pass> createAnalyzeISAXPass();

} // namespace shortnail
} // namespace mlir

#endif // SHORTNAIL_CONVERSION_ANALYZEISAX_H
