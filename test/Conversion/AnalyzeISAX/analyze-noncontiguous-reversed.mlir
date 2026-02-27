// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Instruction with a reversed non-contiguous encoding field.
// Reversed means the CoreDSL field declaration used imm8[0:3] instead of
// imm8[3:0]. Treenail normalizes start >= end and sets reversed=1.
// When reversed, the encoding MSB maps to field bit `end` and LSB to `start`.
// So field_ranges are (end, start) instead of the normal (start, end).

// CHECK: extension: "testisax"
// CHECK:   - name: "revtest"
// CHECK:     fields:
// CHECK:       - name: "imm8"
// CHECK:         ranges:
// CHECK:           - "31-28"
// CHECK:           - "12-9"
// CHECK:         field_ranges:
// CHECK:           - "0-3"
// CHECK:           - "4-7"
// CHECK:         bits: 8
// CHECK:         type: "mask"

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  // imm8 reversed: CoreDSL says imm8[0:3] :: ... :: imm8[4:7]
  // Encoding: imm8_3_0(4) :: 0000000(7) :: rs1(5) :: 000(3) :: imm8_7_4(4) :: 0001011(7) :: 00(2) = 32
  coredsl.instruction @REVTEST {lil.enc_immediates = [[["%TREENAIL_WAS_HERE_imm8_3_0", 3, 0, 1, "imm8"], ["%TREENAIL_WAS_HERE_imm8_7_4", 7, 4, 1, "imm8"]], [["%rs1", 4, 0, 0, "rs1"]]]} (%TREENAIL_WAS_HERE_imm8_3_0 : ui4, "0000000", %rs1 : ui5, "000", %TREENAIL_WAS_HERE_imm8_7_4 : ui4, "000101100") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    coredsl.end
  }
}
}
