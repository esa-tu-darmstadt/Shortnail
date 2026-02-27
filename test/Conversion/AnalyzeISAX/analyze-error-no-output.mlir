// RUN: not shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax)' %s 2>&1 | FileCheck %s

// Missing 'output' option should produce an error.
// CHECK: 'output' option is required

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.instruction @NOP ("0000000", "00000", "00000", "000", "00000", "1111011") {
    coredsl.end
  }
}
}
