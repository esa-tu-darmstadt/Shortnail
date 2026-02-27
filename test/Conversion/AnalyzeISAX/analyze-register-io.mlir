// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Instruction where rd and rs1 are the same field (constrained/tied operand).

// CHECK: extension: "testisax"
// CHECK:   - name: "inc"
// CHECK:     fields:
// CHECK:       - name: "rd"
// CHECK:         bits: 5
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:         is_output: true
// CHECK:         constrained: true

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.instruction @INC {lil.enc_immediates = [[["%rd", 4, 0, 0, "rd"]]]} ("0000000", "00000", "00000", "000", %rd : ui5, "1111011") {
    %0 = coredsl.get @X[%rd : ui5] : ui32
    %1 = hwarith.constant 1 : ui32
    %2 = coredsl.or %0, %1 : ui32, ui32
    coredsl.set @X[%rd : ui5] = %2 : ui32
    coredsl.end
  }
}
}
