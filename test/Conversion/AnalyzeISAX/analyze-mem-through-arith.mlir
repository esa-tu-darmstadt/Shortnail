// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Tests that mem_output is detected when a memory load flows through
// arithmetic operations before being written to a register.
// This exercises the expanded valueComesFromMem() worklist.

// CHECK: extension: "testisax"
// CHECK:   - name: "loadxor"
// CHECK:     side_effects: "read"
// CHECK:     fields:
// CHECK:       - name: "rs2"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:         is_output: false
// CHECK:       - name: "rs1"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:         mem_input: true
// CHECK:       - name: "rd"
// CHECK:         type: "register"
// CHECK:         is_output: true
// CHECK:         mem_output: true

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8

  // Load from memory, XOR with rs2, shift, then write to rd.
  // The chain is: get @MEM → cast → xor → shift_left → set @X[rd]
  coredsl.instruction @LOADXOR {lil.enc_immediates = [[["%rs2", 4, 0, 0, "rs2"]], [["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]} ("0000111", %rs2 : ui5, %rs1 : ui5, "000", %rd : ui5, "0001011") {
    %addr = coredsl.get @X[%rs1 : ui5] : ui32
    %byte = coredsl.get @MEM[%addr: ui32] : ui8
    %ext = coredsl.cast %byte : ui8 to ui32
    %mask = coredsl.get @X[%rs2 : ui5] : ui32
    %xored = coredsl.xor %ext, %mask : ui32, ui32
    %shamt = hwarith.constant 8 : ui4
    %shifted = coredsl.shift_left %xored, %shamt : ui32, ui4
    coredsl.set @X[%rd : ui5] = %shifted : ui32
    coredsl.end
  }
}
}
