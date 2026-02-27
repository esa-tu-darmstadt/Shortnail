// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Instruction with a mask-type field (encoding bits unused by the behavior).

// CHECK: extension: "testisax"
// CHECK:   - name: "nop5"
// CHECK:     fields:
// CHECK:       - name: "unused"
// CHECK:         bits: 5
// CHECK:         type: "mask"
// CHECK:       - name: "rs1"
// CHECK:         bits: 5
// CHECK:         type: "register"

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.instruction @NOP5 {lil.enc_immediates = [[["%unused", 4, 0, 0, "unused"]], [["%rs1", 4, 0, 0, "rs1"]]]} ("0000000", %unused : ui5, %rs1 : ui5, "000", "00000", "1111011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    coredsl.end
  }
}
}
