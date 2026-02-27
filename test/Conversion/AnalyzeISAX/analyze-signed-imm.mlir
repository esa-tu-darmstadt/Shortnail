// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Tests signed immediate detection: an immediate that gets cast to a signed
// type should be marked signed: true.

// CHECK: extension: "testisax"
// CHECK:   - name: "addsimm"
// CHECK:     side_effects: "none"
// CHECK:     fields:
// CHECK:       - name: "simm7"
// CHECK:         bits: 7
// CHECK:         type: "immediate"
// CHECK:         signed: true
// CHECK:       - name: "rs1"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:       - name: "rd"
// CHECK:         type: "register"
// CHECK:         is_output: true

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32
  coredsl.instruction @ADDSIMM {lil.enc_immediates = [[["%simm7", 6, 0, 0, "simm7"]], [["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]} (%simm7 : ui7, "00000", %rs1 : ui5, "000", %rd : ui5, "1111011") {
    %a = coredsl.get @X[%rs1 : ui5] : ui32
    %ext = coredsl.cast %simm7 : ui7 to si7
    %ext32 = coredsl.cast %ext : si7 to si32
    %a_signed = coredsl.cast %a : ui32 to si32
    %sum33 = hwarith.add %a_signed, %ext32 : (si32, si32) -> si33
    %sum = coredsl.cast %sum33 : si33 to ui32
    coredsl.set @X[%rd : ui5] = %sum : ui32
    coredsl.end
  }
}
}
