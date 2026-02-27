// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Test side effect detection: memory read, memory write, local register, PC write.

// CHECK: extension: "testisax"

// Pure instruction (no side effects).
// CHECK:   - name: "pure"
// CHECK:     side_effects: "none"

// Memory load -> read side effect, mem_input on rs1, mem_output on rd.
// CHECK:   - name: "memload"
// CHECK:     side_effects: "read"
// CHECK:     fields:
// CHECK:       - name: "rs1"
// CHECK:         type: "register"
// CHECK:         mem_input: true
// CHECK:       - name: "rd"
// CHECK:         type: "register"
// CHECK:         mem_output: true

// Memory store -> write side effect.
// CHECK:   - name: "memstore"
// CHECK:     side_effects: "write"

// Local register write -> write side effect.
// CHECK:   - name: "localwrite"
// CHECK:     side_effects: "write"

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8
  coredsl.register local @SCRATCH : ui32

  // Pure: register-to-register only
  coredsl.instruction @PURE {lil.enc_immediates = [[["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]} ("0000000", "00000", %rs1 : ui5, "000", %rd : ui5, "0001011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    coredsl.set @X[%rd : ui5] = %0 : ui32
    coredsl.end
  }

  // Memory load: read from MEM
  coredsl.instruction @MEMLOAD {lil.enc_immediates = [[["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]} ("0000001", "00000", %rs1 : ui5, "000", %rd : ui5, "0001011") {
    %addr = coredsl.get @X[%rs1 : ui5] : ui32
    %val = coredsl.get @MEM[%addr: ui32] : ui8
    %ext = coredsl.cast %val : ui8 to ui32
    coredsl.set @X[%rd : ui5] = %ext : ui32
    coredsl.end
  }

  // Memory store: write to MEM
  coredsl.instruction @MEMSTORE {lil.enc_immediates = [[["%rs2", 4, 0, 0, "rs2"]], [["%rs1", 4, 0, 0, "rs1"]]]} ("0000010", %rs2 : ui5, %rs1 : ui5, "000", "00000", "0001011") {
    %addr = coredsl.get @X[%rs1 : ui5] : ui32
    %val = coredsl.get @X[%rs2 : ui5] : ui32
    %trunc = coredsl.cast %val : ui32 to ui8
    coredsl.set @MEM[%addr: ui32] = %trunc : ui8
    coredsl.end
  }

  // Local register write
  coredsl.instruction @LOCALWRITE {lil.enc_immediates = [[["%rs1", 4, 0, 0, "rs1"]]]} ("0000011", "00000", %rs1 : ui5, "000", "00000", "0001011") {
    %val = coredsl.get @X[%rs1 : ui5] : ui32
    coredsl.set @SCRATCH = %val : ui32
    coredsl.end
  }
}
}
