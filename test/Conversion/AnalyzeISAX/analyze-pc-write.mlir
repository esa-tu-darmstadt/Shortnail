// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Writing to core_pc sets is_jump = true.

// CHECK: extension: "testisax"
// CHECK:   - name: "jmpabs"
// CHECK:     is_jump: true
// CHECK:     side_effects: "none"
// CHECK:     fields:
// CHECK:       - name: "rs1"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:         is_output: false

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.register core_pc @PC : ui32
  coredsl.instruction @JMPABS {lil.enc_immediates = [[["%rs1", 4, 0, 0, "rs1"]]]} ("0001000", "00000", %rs1 : ui5, "000", "00000", "0001011") {
    %addr = coredsl.get @X[%rs1 : ui5] : ui32
    coredsl.set @PC = %addr : ui32
    coredsl.end
  }
}
}
