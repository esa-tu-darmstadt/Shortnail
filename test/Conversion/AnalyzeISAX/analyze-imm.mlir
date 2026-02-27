// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Instruction with a mask-type field (encoding bits unused by the behavior).

// CHECK: extension: "testisax"
// CHECK:   - name: "nop5"
// CHECK:     fields:
// CHECK:       - name: "used"
// CHECK:         bits: 5
// CHECK:         type: "immediate"
// CHECK:         signed: false
// CHECK:       - name: "rs1"
// CHECK:         bits: 5
// CHECK:         type: "register"

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.instruction @NOP5 {lil.enc_immediates = [[["%used", 4, 0, 0, "used"]], [["%rs1", 4, 0, 0, "rs1"]]]} ("0000000", %used : ui5, %rs1 : ui5, "000", "00000", "1111011") {
    %c0 = hwarith.constant 0 : ui1
    %cond = hwarith.icmp eq %used, %c0 : ui5, ui1
    scf.if %cond {
      %0 = coredsl.get @X[%rs1 : ui5] : ui32
    }
    coredsl.end
  }
}
}
