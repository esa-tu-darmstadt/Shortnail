// RUN: shortnail-opt %s -coredsl-switch-to-if -canonicalize | shortnail-opt | FileCheck %s

coredsl.isax "SwitchStmt" {
// CHECK: coredsl.isax "SwitchStmt"
  coredsl.register core_x @X[32] : ui32
  coredsl.register local @PC : ui32
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8
  coredsl.register local const @MY_CONST = 3 : ui32
  coredsl.instruction @Simple {lil.enc_immediates = [[["%TREENAIL_WAS_HERE_rs2_4_0", 4, 0, 0, "rs2"]], [["%TREENAIL_WAS_HERE_rs1_4_0", 4, 0, 0, "rs1"]], [["%TREENAIL_WAS_HERE_rd_4_0", 4, 0, 0, "rd"]]]} ("0000000", %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "001", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0101011") {

// CHECK: %0 = hwarith.constant 1 : ui1
// CHECK: %1 = hwarith.constant 7 : ui3
// CHECK: %c3 = arith.constant 3 : index
// CHECK: %2 = hwarith.constant 5 : ui3
// CHECK: %c2 = arith.constant 2 : index
// CHECK: %3 = hwarith.constant 10 : ui4
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %4 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
// CHECK: %5 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK: %6 = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK: %7 = coredsl.cast %6 : ui5 to ui32
// CHECK: %8 = coredsl.get @MEM[%7 : ui32] : ui8
// CHECK: %9 = coredsl.cast %8 : ui8 to ui32
// CHECK: %10 = coredsl.get @MEM[1:0] : ui16
// CHECK: %11 = coredsl.get @MEM[2] : ui8
// CHECK: %12 = hwarith.cast %4 : (ui5) -> i6
// CHECK: %13 = arith.index_castui %12 : i6 to index
// CHECK: %14 = arith.cmpi eq, %13, %c0 : index
// CHECK: %15:3 = scf.if %14 -> (ui32, ui16, ui8) {
// CHECK:   scf.yield %9, %10, %11 : ui32, ui16, ui8
// CHECK: } else {
// CHECK:   %16 = arith.cmpi eq, %13, %c1 : index
// CHECK:   %17:3 = scf.if %16 -> (ui32, ui16, ui8) {
// CHECK:     %18 = coredsl.cast %3 : ui4 to ui32
// CHECK:     scf.yield %18, %10, %11 : ui32, ui16, ui8
// CHECK:   } else {
// CHECK:     %18 = arith.cmpi eq, %13, %c2 : index
// CHECK:     %19:3 = scf.if %18 -> (ui32, ui16, ui8) {
// CHECK:       %20 = coredsl.cast %2 : ui3 to ui16
// CHECK:       scf.yield %9, %20, %11 : ui32, ui16, ui8
// CHECK:     } else {
// CHECK:       %20 = arith.cmpi eq, %13, %c3 : index
// CHECK:       %21:3 = scf.if %20 -> (ui32, ui16, ui8) {
// CHECK:         %22 = coredsl.cast %1 : ui3 to ui8
// CHECK:         scf.yield %9, %10, %22 : ui32, ui16, ui8
// CHECK:       } else {
// CHECK:         %22 = coredsl.cast %0 : ui1 to ui32
// CHECK:         %23 = coredsl.cast %0 : ui1 to ui16
// CHECK:         %24 = coredsl.cast %0 : ui1 to ui8
// CHECK:         scf.yield %22, %23, %24 : ui32, ui16, ui8
// CHECK:       }
// CHECK:       scf.yield %21#0, %21#1, %21#2 : ui32, ui16, ui8
// CHECK:     }
// CHECK:     scf.yield %19#0, %19#1, %19#2 : ui32, ui16, ui8
// CHECK:   }
// CHECK:   scf.yield %17#0, %17#1, %17#2 : ui32, ui16, ui8
// CHECK: }
// CHECK: coredsl.set @X[%5 : ui5] = %15#0 : ui32
// CHECK: coredsl.set @MEM[1:0] = %15#1 : ui16
// CHECK: coredsl.set @MEM[2] = %15#2 : ui8

    %rs2 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %1 = coredsl.cast %rd : ui5 to ui32
    %0 = coredsl.get @MEM[%1 : ui32] : ui8
    %2 = coredsl.cast %0 : ui8 to ui32
    %3 = coredsl.get @MEM[1:0] : ui16
    %4 = coredsl.get @MEM[2] : ui8
    %5 = hwarith.cast %rs2 : (ui5) -> i6
    cf.switch %5 : i6, [
      default: ^default_0,
      0: ^case_0_1,
      1: ^case_1_2,
      2: ^case_2_3,
      3: ^case_3_4
    ]
    ^case_0_1():
      cf.br ^switch_end_5(%2, %3, %4 : ui32, ui16, ui8)
    ^case_1_2():
      %6 = hwarith.constant 10 : ui4
      %7 = coredsl.cast %6 : ui4 to ui32
      cf.br ^switch_end_5(%7, %3, %4 : ui32, ui16, ui8)
    ^case_2_3():
      %8 = hwarith.constant 5 : ui3
      %9 = coredsl.cast %8 : ui3 to ui16
      cf.br ^switch_end_5(%2, %9, %4 : ui32, ui16, ui8)
    ^case_3_4():
      %10 = hwarith.constant 7 : ui3
      %11 = coredsl.cast %10 : ui3 to ui8
      cf.br ^switch_end_5(%2, %3, %11 : ui32, ui16, ui8)
    ^default_0():
      %12 = hwarith.constant 1 : ui1
      %13 = coredsl.cast %12 : ui1 to ui32
      %14 = hwarith.constant 1 : ui1
      %15 = coredsl.cast %14 : ui1 to ui16
      %16 = hwarith.constant 1 : ui1
      %17 = coredsl.cast %16 : ui1 to ui8
      cf.br ^switch_end_5(%13, %15, %17 : ui32, ui16, ui8)
    ^switch_end_5(%18: ui32, %19: ui16, %20: ui8):
    coredsl.set @X[%rs1 : ui5] = %18 : ui32
    coredsl.set @MEM[1:0] = %19 : ui16
    coredsl.set @MEM[2] = %20 : ui8
    coredsl.end
  }
  coredsl.instruction @NestedSwitch {lil.enc_immediates = [[["%TREENAIL_WAS_HERE_rs2_4_0", 4, 0, 0, "rs2"]], [["%TREENAIL_WAS_HERE_rs1_4_0", 4, 0, 0, "rs1"]], [["%TREENAIL_WAS_HERE_rd_4_0", 4, 0, 0, "rd"]]]} ("0000000", %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "001", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0101011") {

// CHECK: %0 = hwarith.constant 7 : ui3
// CHECK: %c3 = arith.constant 3 : index
// CHECK: %1 = hwarith.constant 5 : ui3
// CHECK: %2 = hwarith.constant 10 : ui4
// CHECK: %c1 = arith.constant 1 : index
// CHECK: %3 = hwarith.constant 2 : ui2
// CHECK: %c2 = arith.constant 2 : index
// CHECK: %4 = hwarith.constant 1 : ui1
// CHECK: %c16 = arith.constant 16 : index
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %5 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
// CHECK: %6 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK: %7 = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK: %8 = coredsl.cast %7 : ui5 to ui32
// CHECK: %9 = coredsl.get @MEM[%8 : ui32] : ui8
// CHECK: %10 = coredsl.cast %9 : ui8 to ui32
// CHECK: %11 = coredsl.get @MEM[1:0] : ui16
// CHECK: %12 = coredsl.get @MEM[2] : ui8
// CHECK: %13 = hwarith.cast %5 : (ui5) -> i6
// CHECK: %14 = arith.index_castui %13 : i6 to index
// CHECK: %15 = arith.cmpi eq, %14, %c0 : index
// CHECK: %16:3 = scf.if %15 -> (ui16, ui32, ui8) {
// CHECK:   %17 = hwarith.cast %7 : (ui5) -> i6
// CHECK:   %18 = arith.index_castui %17 : i6 to index
// CHECK:   %19 = arith.cmpi eq, %18, %c16 : index
// CHECK:   %20 = scf.if %19 -> (ui16) {
// CHECK:     %21 = hwarith.sub %11, %4 : (ui16, ui1) -> si17
// CHECK:     %22 = coredsl.cast %21 : si17 to ui16
// CHECK:     scf.yield %22 : ui16
// CHECK:   } else {
// CHECK:     %21 = arith.cmpi eq, %18, %c2 : index
// CHECK:     %22 = scf.if %21 -> (ui16) {
// CHECK:       %23 = hwarith.add %11, %4 : (ui16, ui1) -> ui17
// CHECK:       %24 = coredsl.cast %23 : ui17 to ui16
// CHECK:       scf.yield %24 : ui16
// CHECK:     } else {
// CHECK:       %23 = hwarith.add %11, %3 : (ui16, ui2) -> ui17
// CHECK:       %24 = coredsl.cast %23 : ui17 to ui16
// CHECK:       scf.yield %24 : ui16
// CHECK:     }
// CHECK:     scf.yield %22 : ui16
// CHECK:   }
// CHECK:   scf.yield %20, %10, %12 : ui16, ui32, ui8
// CHECK: } else {
// CHECK:   %17 = arith.cmpi eq, %14, %c1 : index
// CHECK:   %18:3 = scf.if %17 -> (ui16, ui32, ui8) {
// CHECK:     %19 = coredsl.cast %2 : ui4 to ui32
// CHECK:     scf.yield %11, %19, %12 : ui16, ui32, ui8
// CHECK:   } else {
// CHECK:     %19 = arith.cmpi eq, %14, %c2 : index
// CHECK:     %20:2 = scf.if %19 -> (ui16, ui8) {
// CHECK:       %21 = coredsl.cast %1 : ui3 to ui16
// CHECK:       scf.yield %21, %12 : ui16, ui8
// CHECK:     } else {
// CHECK:       %21 = arith.cmpi eq, %14, %c3 : index
// CHECK:       %22 = scf.if %21 -> (ui8) {
// CHECK:         %23 = coredsl.cast %0 : ui3 to ui8
// CHECK:         scf.yield %23 : ui8
// CHECK:       } else {
// CHECK:         scf.yield %12 : ui8
// CHECK:       }
// CHECK:       scf.yield %11, %22 : ui16, ui8
// CHECK:     }
// CHECK:     scf.yield %20#0, %10, %20#1 : ui16, ui32, ui8
// CHECK:   }
// CHECK:   scf.yield %18#0, %18#1, %18#2 : ui16, ui32, ui8
// CHECK: }
// CHECK: coredsl.set @X[%6 : ui5] = %16#1 : ui32
// CHECK: coredsl.set @MEM[1:0] = %16#0 : ui16
// CHECK: coredsl.set @MEM[2] = %16#2 : ui8

    %rs2 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %1 = coredsl.cast %rd : ui5 to ui32
    %0 = coredsl.get @MEM[%1 : ui32] : ui8
    %2 = coredsl.cast %0 : ui8 to ui32
    %3 = coredsl.get @MEM[1:0] : ui16
    %4 = coredsl.get @MEM[2] : ui8
    %5 = hwarith.cast %rs2 : (ui5) -> i6
    cf.switch %5 : i6, [
      default: ^default_0,
      0: ^case_0_1,
      1: ^case_1_2,
      2: ^case_2_3,
      3: ^case_3_4
    ]
    ^case_0_1():
      %6 = hwarith.cast %rd : (ui5) -> i6
      cf.switch %6 : i6, [
        default: ^default_6,
        16: ^case_16_7,
        2: ^case_2_8
      ]
      ^case_16_7():
        %7 = hwarith.constant 1 : ui1
        %8 = hwarith.sub %3, %7 : (ui16, ui1) -> si17
        %9 = coredsl.cast %8 : si17 to ui16
        cf.br ^switch_end_9(%9 : ui16)
      ^case_2_8():
        %10 = hwarith.constant 1 : ui1
        %11 = hwarith.add %3, %10 : (ui16, ui1) -> ui17
        %12 = coredsl.cast %11 : ui17 to ui16
        cf.br ^switch_end_9(%12 : ui16)
      ^default_6():
        %13 = hwarith.constant 2 : ui2
        %14 = hwarith.add %3, %13 : (ui16, ui2) -> ui17
        %15 = coredsl.cast %14 : ui17 to ui16
        cf.br ^switch_end_9(%15 : ui16)
      ^switch_end_9(%16: ui16):
      cf.br ^switch_end_5(%16, %2, %4 : ui16, ui32, ui8)
    ^case_1_2():
      %17 = hwarith.constant 10 : ui4
      %18 = coredsl.cast %17 : ui4 to ui32
      cf.br ^switch_end_5(%3, %18, %4 : ui16, ui32, ui8)
    ^case_2_3():
      %19 = hwarith.constant 5 : ui3
      %20 = coredsl.cast %19 : ui3 to ui16
      cf.br ^switch_end_5(%20, %2, %4 : ui16, ui32, ui8)
    ^case_3_4():
      %21 = hwarith.constant 7 : ui3
      %22 = coredsl.cast %21 : ui3 to ui8
      cf.br ^switch_end_5(%3, %2, %22 : ui16, ui32, ui8)
    ^default_0():
      cf.br ^switch_end_5(%3, %2, %4 : ui16, ui32, ui8)
    ^switch_end_5(%23: ui16, %24: ui32, %25: ui8):
    coredsl.set @X[%rs1 : ui5] = %24 : ui32
    coredsl.set @MEM[1:0] = %23 : ui16
    coredsl.set @MEM[2] = %25 : ui8
    coredsl.end
  }
}

