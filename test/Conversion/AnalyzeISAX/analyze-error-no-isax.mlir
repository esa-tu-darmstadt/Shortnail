// RUN: not shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s 2>&1 | FileCheck %s

// Empty module (no ISAXOp) should produce an error.
// CHECK: expected exactly 1 coredsl.ISAXOp, found 0

module {
}
