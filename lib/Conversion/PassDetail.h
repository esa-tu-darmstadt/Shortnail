//===- PassDetail.h - Conversion pass class details -------------*- C++ -*-===//
//
// Copyright 2021 Embedded Systems and Applications Group
//                Department of Computer Science
//                Technical University of Darmstadt, Germany
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_PASSDETAIL_H
#define CONVERSION_PASSDETAIL_H

#include "shortnail/Dialect/CoreDSL/CoreDSLDialect.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace hw {
class HWDialect;
} // namespace hw

namespace hwarith {
class HWArithDialect;
} // namespace hwarith

namespace comb {
class CombDialect;
} // namespace comb

namespace seq {
class SeqDialect;
} // namespace seq

namespace sv {
class SVDialect;
} // namespace sv
} // namespace circt

namespace mlir {
class StandardOpsDialect;
class ModuleOp;

namespace ub {
class UBDialect;
} // namespace ub

namespace memref {
class MemRefDialect;
} // namespace memref

namespace func {
class FuncDialect;
} // namespace func

namespace coredsl {
class ISAXOp;
} // namespace coredsl

namespace shortnail {

#define GEN_PASS_CLASSES
#include "shortnail/Conversion/Passes.h.inc"

} // namespace shortnail
} // namespace mlir

#endif // CONVERSION_PASSDETAIL_H
