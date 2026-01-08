//===- Passes.h - Conversion pass registration ------------------*- C++ -*-===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#ifndef SHORTNAIL_CONVERSION_PASSES_H
#define SHORTNAIL_CONVERSION_PASSES_H

#include "shortnail/Conversion/CoreDSLToPy.h"
#include "shortnail/Conversion/MergeISAX.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace shortnail {

#define GEN_PASS_REGISTRATION
#include "shortnail/Conversion/Passes.h.inc"

} // namespace shortnail
} // namespace mlir

#endif // SHORTNAIL_CONVERSION_PASSES_H
