// RUN: shortnail-opt '--pass-pipeline=builtin.module(inline,analyze-isax{output=%t})' %s && FileCheck %s --input-file=%t

// INIT_RF_42-style: loop index is not an encoding field → clobbers_all_regs.
// CHECK: - name: "initrf42"
// CHECK:   side_effects: "none"
// CHECK-NOT: clobbers_all_regs
// CHECK:   clobbers_regs: ["x0", "x1",

// Encoding-field GPR write — should NOT be flagged.
// CHECK: - name: "pure"
// CHECK-NOT: clobbers_all_regs
// CHECK-NOT: clobbers_regs

// Pseudo Encoding-field GPR write — mixed constant + encoding arg → clobbers_all_regs.
// CHECK: - name: "fakegprwrite"
// CHECK:   clobbers_all_regs: true

// Uses only encoding fields, but different fields to construct the address -> we cant use the output flag for a certain field
// CHECK: - name: "fakepure"
// CHECK:   clobbers_all_regs: true

// Constant-index GPR write via from/to attribute (no base operand) → specific clobber.
// CHECK: - name: "constidxwrite"
// CHECK-NOT: clobbers_all_regs
// CHECK:   clobbers_regs: ["x1", "x27"]

// Range GPR write (e.g. @X[5:3]) — registers 3, 4, 5 are known.
// CHECK: - name: "constrangewrite1"
// CHECK-NOT: clobbers_all_regs
// CHECK:   clobbers_regs: ["x3", "x4", "x5"]

// Range GPR write (e.g. @X[3:5]) — registers 3, 4, 5 are known.
// CHECK: - name: "constrangewrite2"
// CHECK-NOT: clobbers_all_regs
// CHECK:   clobbers_regs: ["x3", "x4", "x5"]

// Range GPR write (e.g. @X[4, 3:5]) — registers 3, 4, 5 are known.
// CHECK: - name: "constrangewrite3"
// CHECK-NOT: clobbers_all_regs
// CHECK:   clobbers_regs: ["x7", "x8", "x9"]


module {
coredsl.isax "testisax" {
  coredsl.register core_x @X[32] : ui32

  coredsl.instruction @INITRF42 {lil.enc_immediates = []} ("00000000000000000000000001001011") {
    %lo = hw.constant 0 : i7
    %hi = hw.constant 32 : i7
    %step = hw.constant 1 : i7
    %c0 = hwarith.constant 0 : ui32
    scf.for unsigned %i = %lo to %hi step %step : i7 {
      %idx = hwarith.cast %i : (i7) -> ui5
      coredsl.set @X[%idx : ui5] = %c0 : ui32
    }
    coredsl.end
  }

  coredsl.instruction @PURE {lil.enc_immediates = [[["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]}
      ("0000000", "00000", %rs1 : ui5, "000", %rd : ui5, "0001011") {
    %val = coredsl.get @X[%rs1 : ui5] : ui32
    coredsl.set @X[%rd : ui5] = %val : ui32
    coredsl.end
  }

  coredsl.instruction @FAKE_GPR_WRITE {lil.enc_immediates = [[["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]}
      ("0000001", "00000", %rs1 : ui5, "000", %rd : ui5, "0001011") {
    %padding = hwarith.cast %rs1 : (ui5) -> ui1

    %lo = hw.constant 0 : i6
    %hi = hw.constant 16 : i6
    %step = hw.constant 1 : i6
    %c0 = hwarith.constant 0 : ui32
    scf.for unsigned %i = %lo to %hi step %step : i6 {
      %idx_ = hwarith.cast %i : (i6) -> ui4
      %idx = coredsl.concat %padding, %idx_ : ui1, ui4
      coredsl.set @X[%idx : ui5] = %c0 : ui32
    }
    coredsl.end
  }

  coredsl.instruction @FAKE_PURE {lil.enc_immediates = [[["%rs1", 4, 0, 0, "rs1"]], [["%rd", 4, 0, 0, "rd"]]]}
      ("0000010", "00000", %rs1 : ui5, "000", %rd : ui5, "0001011") {
    %i1 = coredsl.cast %rs1 : ui5 to ui3
    %i2 = coredsl.cast %rd : ui5 to ui2
    %idx = coredsl.concat %i1, %i2 : ui3, ui2
    %c0 = hwarith.constant 0 : ui32
    coredsl.set @X[%idx : ui5] = %c0 : ui32
    coredsl.end
  }

  coredsl.instruction @CONST_IDX_WRITE {lil.enc_immediates = []}
      ("00000000000000000000000011001011") {
    %c0 = hwarith.constant 0 : ui32
    coredsl.set @X[27] = %c0 : ui32
    coredsl.set @X[1] = %c0 : ui32
    coredsl.end
  }

  coredsl.instruction @CONST_RANGE_WRITE_1 {lil.enc_immediates = []}
      ("00000000000000000000000111001011") {
    %c0 = hwarith.constant 0 : ui96
    coredsl.set @X[5:3] = %c0 : ui96
    coredsl.end
  }
  coredsl.instruction @CONST_RANGE_WRITE_2 {lil.enc_immediates = []}
      ("00000000000000000000000111001011") {
    %c0 = hwarith.constant 0 : ui96
    coredsl.set @X[3:5] = %c0 : ui96
    coredsl.end
  }

  coredsl.instruction @CONST_RANGE_WRITE_3 {lil.enc_immediates = []}
      ("00000000000000000000000111001011") {
    %idx = hwarith.constant 4 : ui5
    %c0 = hwarith.constant 0 : ui96
    coredsl.set @X[%idx : ui5, 3:5] = %c0 : ui96
    coredsl.end
  }
}
}
