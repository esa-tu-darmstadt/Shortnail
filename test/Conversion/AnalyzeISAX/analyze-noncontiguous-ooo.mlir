// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Instruction with non-contiguous encoding field where bits are OUT OF ORDER.
// Modeled after BRIMM's cv_beqimm encoding:
//   imm12[11:11] :: imm12[9:4] :: imm5[4:0] :: rs1[4:0] :: 3'b100 :: imm12[3:0] :: imm12[10:10] :: 7'b0001011
//
// The key test: field_ranges must reflect the CoreDSL bit indices (11, 9-4, 3-0, 10),
// NOT sequential packing (11, 10-5, 4-1, 0).

// CHECK: extension: "testisax"
// CHECK:   - name: "cvbeqimm"
// CHECK:     fields:
// CHECK:       - name: "imm12"
// CHECK:         ranges:
// CHECK:           - "31-31"
// CHECK:           - "30-25"
// CHECK:           - "11-8"
// CHECK:           - "7-7"
// CHECK:         field_ranges:
// CHECK:           - "11-11"
// CHECK:           - "9-4"
// CHECK:           - "3-0"
// CHECK:           - "10-10"
// CHECK:         bits: 12
// CHECK:         type: "immediate"
// CHECK:         signed: true
// CHECK:       - name: "imm5"
// CHECK:         range: "24-20"
// CHECK:         bits: 5
// CHECK:         type: "mask"
// CHECK:       - name: "rs1"
// CHECK:         range: "19-15"
// CHECK:         bits: 5
// CHECK:         type: "register"
// CHECK:         reg_file: "X"
// CHECK:         reg_width: 32
// CHECK:         is_input: true

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.register core_pc @PC : ui32
  // cv_beqimm: branch to PC + imm12 if X[rs1] == imm5
  // Encoding: imm12[11:11] :: imm12[9:4] :: imm5[4:0] :: rs1[4:0] :: 3'b100 :: imm12[3:0] :: imm12[10:10] :: 7'b0001011
  coredsl.instruction @CV_BEQIMM {lil.enc_immediates = [[["%TREENAIL_WAS_HERE_imm12_11_11", 11, 11, 0, "imm12"], ["%TREENAIL_WAS_HERE_imm12_9_4", 9, 4, 0, "imm12"], ["%TREENAIL_WAS_HERE_imm12_3_0", 3, 0, 0, "imm12"], ["%TREENAIL_WAS_HERE_imm12_10_10", 10, 10, 0, "imm12"]], [["%TREENAIL_WAS_HERE_imm5_4_0", 4, 0, 0, "imm5"]], [["%rs1", 4, 0, 0, "rs1"]]]} (%TREENAIL_WAS_HERE_imm12_11_11 : si1, %TREENAIL_WAS_HERE_imm12_9_4 : si6, %TREENAIL_WAS_HERE_imm5_4_0 : ui5, %rs1 : ui5, "100", %TREENAIL_WAS_HERE_imm12_3_0 : si4, %TREENAIL_WAS_HERE_imm12_10_10 : si1, "0001011") {
    // Read X[rs1]
    %regval = coredsl.get @X[%rs1 : ui5] : ui32
    // Use imm5 in a comparison (makes it an immediate, not mask)
    %imm5_wide = coredsl.cast %TREENAIL_WAS_HERE_imm5_4_0 : ui5 to ui32
    %cmp = hwarith.icmp eq %regval, %imm5_wide : ui32, ui32
    // Use imm12 as branch offset (makes it an immediate + triggers is_jump)
    %imm12_wide = coredsl.cast %TREENAIL_WAS_HERE_imm12_11_11 : si1 to si32
    %pc = coredsl.get @PC : ui32
    %newpc = hwarith.add %pc, %imm12_wide : (ui32, si32) -> si34
    %newpc_trunc = coredsl.cast %newpc : si34 to ui32
    coredsl.set @PC = %newpc_trunc : ui32
    coredsl.end
  }
}
}
