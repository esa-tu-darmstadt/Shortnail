// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Reads from const local registers (LUTs) must NOT count as side effects.
// Writes to non-const local registers must count as write side effects.

// CHECK: extension: "testisax"

// Instruction reading from a const LUT — no side effects.
// CHECK:   - name: "lutread"
// CHECK:     side_effects: "none"

// Instruction writing to a non-const local register — write side effect.
// CHECK:   - name: "scratchwrite"
// CHECK:     side_effects: "write"

// Instruction reading from const LUT and writing to non-const local — write only.
// CHECK:   - name: "lutandwrite"
// CHECK:     side_effects: "write"

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.register local const @LUT[4] = [10, 20, 30, 40] : ui32
  coredsl.register local @SCRATCH : ui32

  // Pure: reads from const LUT — should have no side effects
  coredsl.instruction @LUTREAD {lil.enc_immediates = [[["%imm2", 1, 0, 0, "imm2"]], [["%rd", 4, 0, 0, "rd"]]]} ("0000000", "000", %imm2 : ui2, %rd : ui5, "000", "00000", "0001011") {
    %0 = coredsl.get @LUT[%imm2 : ui2] : ui32
    coredsl.set @X[%rd : ui5] = %0 : ui32
    coredsl.end
  }

  // Write to non-const local register — write side effect
  coredsl.instruction @SCRATCHWRITE {lil.enc_immediates = [[["%rs1", 4, 0, 0, "rs1"]]]} ("0000001", "00000", %rs1 : ui5, "000", "00000", "0001011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    coredsl.set @SCRATCH = %0 : ui32
    coredsl.end
  }

  // Read from const LUT + write to non-const local — only write side effect
  coredsl.instruction @LUTANDWRITE {lil.enc_immediates = [[["%imm2", 1, 0, 0, "imm2"]], [["%rs1", 4, 0, 0, "rs1"]]]} ("0000010", "000", %imm2 : ui2, %rs1 : ui5, "000", "00000", "0001011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @LUT[%imm2 : ui2] : ui32
    %2 = coredsl.xor %0, %1 : ui32, ui32
    coredsl.set @SCRATCH = %2 : ui32
    coredsl.end
  }
}
}
