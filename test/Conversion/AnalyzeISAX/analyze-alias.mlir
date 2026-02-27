// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Register accessed through an alias should be detected the same as direct access.

// CHECK: extension: "testisax"
// CHECK:   - name: "aliasinc"
// CHECK:     side_effects: "none"
// CHECK:     fields:
// CHECK:       - name: "rd"
// CHECK:         type: "register"
// CHECK:         reg_file: "X"
// CHECK:         is_input: true
// CHECK:         is_output: true
// CHECK:         constrained: true

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.alias @R = @X[31:0]
  coredsl.instruction @ALIASINC {lil.enc_immediates = [[["%rd", 4, 0, 0, "rd"]]]} ("0000000", "00000", "00000", "000", %rd : ui5, "1111011") {
    %0 = coredsl.get @R[%rd : ui5] : ui32
    %1 = hwarith.constant 1 : ui32
    %2 = coredsl.or %0, %1 : ui32, ui32
    coredsl.set @R[%rd : ui5] = %2 : ui32
    coredsl.end
  }
}
}
