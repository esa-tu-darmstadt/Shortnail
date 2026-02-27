// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Instruction with non-contiguous encoding field (B-type style split immediate).
// imm12 is split: bits [31:25] (7 bits) + bits [11:8] (4 bits) + bit [7:7] (1 bit) = 12 bits total.

// CHECK: extension: "testisax"
// CHECK:   - name: "brtest"
// CHECK:     fields:
// CHECK:       - name: "imm12"
// CHECK:         ranges:
// CHECK:           - "31-25"
// CHECK:           - "11-8"
// CHECK:           - "7-7"
// CHECK:         field_ranges:
// CHECK:           - "11-5"
// CHECK:           - "4-1"
// CHECK:           - "0-0"
// CHECK:         bits: 12
// CHECK:         type: "mask"

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.instruction @BRTEST {lil.enc_immediates = [[["%TREENAIL_WAS_HERE_imm12_11_5", 11, 5, 0, "imm12"], ["%TREENAIL_WAS_HERE_imm12_4_1", 4, 1, 0, "imm12"], ["%TREENAIL_WAS_HERE_imm12_0_0", 0, 0, 0, "imm12"]], [["%rs1", 4, 0, 0, "rs1"]]]} (%TREENAIL_WAS_HERE_imm12_11_5 : ui7, "00000", %rs1 : ui5, "000", %TREENAIL_WAS_HERE_imm12_4_1 : ui4, %TREENAIL_WAS_HERE_imm12_0_0 : ui1, "1111011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    coredsl.end
  }
}
}
