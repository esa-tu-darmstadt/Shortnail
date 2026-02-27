// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// Instruction with hwarith arithmetic operations (add, sub, mul, icmp).
// Verifies that these are treated as pure (no side effects) and that
// register I/O is correctly traced through arithmetic operations.

// CHECK: extension: "testisax"

// Pure arithmetic instruction: add + sub + compare.
// CHECK:   - name: "arith"
// CHECK:     side_effects: "none"
// CHECK:     fields:
// CHECK:       - name: "rs2"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:         is_output: false
// CHECK:       - name: "rs1"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:         is_output: false
// CHECK:       - name: "rd"
// CHECK:         type: "register"
// CHECK:         is_input: false
// CHECK:         is_output: true

// Multiply instruction.
// CHECK:   - name: "mulrd"
// CHECK:     side_effects: "none"
// CHECK:     fields:
// CHECK:       - name: "rs2"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:       - name: "rs1"
// CHECK:         type: "register"
// CHECK:         is_input: true
// CHECK:       - name: "rd"
// CHECK:         type: "register"
// CHECK:         is_output: true

module {
coredsl.isax "TestISAX" {
  coredsl.register core_x @X[32] : ui32

  // add + sub + compare chain
  coredsl.instruction @ARITH {lil.enc_immediates = [[["%rs2", 4, 0, 0, "rs2"]], [["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]} ("0000100", %rs2 : ui5, %rs1 : ui5, "000", %rd : ui5, "1111011") {
    %a = coredsl.get @X[%rs1 : ui5] : ui32
    %b = coredsl.get @X[%rs2 : ui5] : ui32
    %sum33 = hwarith.add %a, %b : (ui32, ui32) -> ui33
    %sum = coredsl.cast %sum33 : ui33 to ui32
    %one = hwarith.constant 1 : ui32
    %diff33 = hwarith.sub %sum, %one : (ui32, ui32) -> si33
    %diff = coredsl.cast %diff33 : si33 to ui32
    coredsl.set @X[%rd : ui5] = %diff : ui32
    coredsl.end
  }

  // multiply
  coredsl.instruction @MULRD {lil.enc_immediates = [[["%rs2", 4, 0, 0, "rs2"]], [["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]} ("0000101", %rs2 : ui5, %rs1 : ui5, "000", %rd : ui5, "1111011") {
    %a = coredsl.get @X[%rs1 : ui5] : ui32
    %b = coredsl.get @X[%rs2 : ui5] : ui32
    %prod64 = hwarith.mul %a, %b : (ui32, ui32) -> ui64
    %prod = coredsl.cast %prod64 : ui64 to ui32
    coredsl.set @X[%rd : ui5] = %prod : ui32
    coredsl.end
  }
}
}
