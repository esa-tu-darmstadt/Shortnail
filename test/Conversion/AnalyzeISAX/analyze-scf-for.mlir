// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Instruction with scf.for loop (integer square root algorithm from real ISAX).
// Verifies that register I/O is correctly detected through loop-carried values
// and that hwarith ops inside loops don't produce false side effects.

// CHECK: extension: "testisax"
// CHECK:   - name: "isqrt"
// CHECK:     side_effects: "none"
// CHECK:     fields:
// CHECK:       - name: "rs1"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:         is_output: false
// CHECK:       - name: "rd"
// CHECK:         type: "register"
// CHECK:         is_input: false
// CHECK:         is_output: true

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32

  coredsl.instruction @ISQRT {lil.enc_immediates = [[["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]} ("0000101", "00000", %rs1 : ui5, "000", %rd : ui5, "0001011") {
    %input = coredsl.get @X[%rs1 : ui5] : ui32
    %c1073741824 = hwarith.constant 1073741824 : ui31
    %bit_init = coredsl.cast %c1073741824 : ui31 to ui32
    %c0 = hwarith.constant 0 : ui1
    %res_init = coredsl.cast %c0 : ui1 to ui32
    %lo = hw.constant 0 : i7
    %hi = hw.constant 32 : i7
    %step = hw.constant 1 : i7
    %result:3 = scf.for unsigned %i = %lo to %hi step %step
        iter_args(%num = %input, %res = %res_init, %bit = %bit_init)
        -> (ui32, ui32, ui32) : i7 {
      %sum33 = hwarith.add %res, %bit : (ui32, ui32) -> ui33
      %sum = coredsl.cast %sum33 : ui33 to ui32
      %ge = hwarith.icmp ge %num, %sum : ui32, ui32
      %ge_ui1 = hwarith.cast %ge : (i1) -> ui1
      %ge_i1 = coredsl.cast %ge_ui1 : ui1 to i1
      %new_num, %new_res = scf.if %ge_i1 -> (ui32, ui32) {
        %diff33 = hwarith.sub %num, %sum : (ui32, ui32) -> si33
        %diff = coredsl.cast %diff33 : si33 to ui32
        %bump33 = hwarith.add %sum, %bit : (ui32, ui32) -> ui33
        %bump = coredsl.cast %bump33 : ui33 to ui32
        scf.yield %diff, %bump : ui32, ui32
      } else {
        scf.yield %num, %res : ui32, ui32
      }
      %c1_shift = hwarith.constant 1 : ui1
      %res_shifted = coredsl.shift_left %new_num, %c1_shift : ui32, ui1
      %bit_shifted = coredsl.shift_right %bit, %c1_shift : ui32, ui1
      scf.yield %res_shifted, %new_res, %bit_shifted : ui32, ui32, ui32
    }
    coredsl.set @X[%rd : ui5] = %result#1 : ui32
    coredsl.end
  }
}
}
