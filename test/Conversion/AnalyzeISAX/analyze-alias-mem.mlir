// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Memory accessed through an alias should still detect read side effects
// and mem_input/mem_output flags.

// CHECK: extension: "testisax"
// CHECK:   - name: "aliasload"
// CHECK:     side_effects: "read"
// CHECK:     fields:
// CHECK:       - name: "rs1"
// CHECK:         type: "register"
// CHECK:         mem_input: true
// CHECK:       - name: "rd"
// CHECK:         type: "register"
// CHECK:         mem_output: true

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8
  coredsl.alias @GPRS = @X[31:0]
  coredsl.instruction @ALIASLOAD {lil.enc_immediates = [[["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]} ("0000100", "00000", %rs1 : ui5, "000", %rd : ui5, "0001011") {
    %addr = coredsl.get @GPRS[%rs1 : ui5] : ui32
    %val = coredsl.get @MEM[%addr: ui32] : ui8
    %ext = coredsl.cast %val : ui8 to ui32
    coredsl.set @GPRS[%rd : ui5] = %ext : ui32
    coredsl.end
  }
}
}
