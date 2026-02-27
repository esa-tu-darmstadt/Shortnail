// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Instruction with scf.if control flow. The analysis should still detect
// register I/O and side effects inside conditional branches.

// CHECK: extension: "testisax"

// Conditional memory store — write side effect even though inside scf.if.
// CHECK:   - name: "condstore"
// CHECK:     side_effects: "write"
// CHECK:     fields:
// CHECK:       - name: "rs2"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:       - name: "rs1"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:         mem_input: true
// CHECK:       - name: "rd"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:         is_output: true
// CHECK:         constrained: true

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8

  coredsl.instruction @CONDSTORE {lil.enc_immediates = [[["%rs2", 4, 0, 0, "rs2"]], [["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]} ("0000110", %rs2 : ui5, %rs1 : ui5, "000", %rd : ui5, "0001011") {
    %a = coredsl.get @X[%rs1 : ui5] : ui32
    %b = coredsl.get @X[%rs2 : ui5] : ui32
    %old = coredsl.get @X[%rd : ui5] : ui32
    %zero = hwarith.constant 0 : ui32
    %cmp = hwarith.icmp ne %b, %zero : ui32, ui32
    %cond = hwarith.cast %cmp : (i1) -> ui1
    %cond_i1 = coredsl.cast %cond : ui1 to i1
    %result = scf.if %cond_i1 -> (ui32) {
      // Store b to memory at address a
      %trunc = coredsl.cast %b : ui32 to ui8
      coredsl.set @MEM[%a: ui32] = %trunc : ui8
      // Return old + 1
      %one = hwarith.constant 1 : ui32
      %inc33 = hwarith.add %old, %one : (ui32, ui32) -> ui33
      %inc = coredsl.cast %inc33 : ui33 to ui32
      scf.yield %inc : ui32
    } else {
      scf.yield %old : ui32
    }
    coredsl.set @X[%rd : ui5] = %result : ui32
    coredsl.end
  }
}
}
