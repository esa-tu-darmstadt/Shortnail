//===- MergeISAX.h - Combine multiple ISAXes ---------------------*- C++ -*-==//
//
// Copyright 2024 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#ifndef SHORTNAIL_CONVERSION_MERGEISAX_H
#define SHORTNAIL_CONVERSION_MERGEISAX_H

#include <memory>

namespace mlir {

class Pass;

namespace shortnail {

std::unique_ptr<mlir::Pass> createMergeISAXPass();

} // namespace shortnail
} // namespace mlir

#endif // SHORTNAIL_CONVERSION_MERGEISAX_H
