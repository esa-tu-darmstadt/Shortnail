// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Basic instruction with one register input (rs1) and one immediate (imm7).

// CHECK: extension: "testisax"
// CHECK: instructions:
// CHECK:   - name: "addi7"
// CHECK:     mask: "-------00000-----000-----1111011"
// CHECK:     is_jump: false
// CHECK:     side_effects: "none"
// CHECK:     fields:
// CHECK:       - name: "imm7"
// CHECK:         range: "31-25"
// CHECK:         bits: 7
// CHECK:         type: "immediate"
// CHECK:         signed: false
// CHECK:       - name: "rs1"
// CHECK:         range: "19-15"
// CHECK:         bits: 5
// CHECK:         type: "register"
// CHECK:         reg_file: "X"
// CHECK:         reg_width: 32
// CHECK:         is_input: true
// CHECK:         is_output: false
// CHECK:       - name: "rd"
// CHECK:         range: "11-7"
// CHECK:         bits: 5
// CHECK:         type: "register"
// CHECK:         reg_file: "X"
// CHECK:         reg_width: 32
// CHECK:         is_input: false
// CHECK:         is_output: true

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.instruction @ADDI7 {lil.enc_immediates = [[["%imm7", 6, 0, 0, "imm7"]], [["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]} (%imm7 : ui7, "00000", %rs1 : ui5, "000", %rd : ui5, "1111011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.cast %imm7 : ui7 to ui32
    %2 = coredsl.or %0, %1 : ui32, ui32
    coredsl.set @X[%rd : ui5] = %2 : ui32
    coredsl.end
  }
}
}
