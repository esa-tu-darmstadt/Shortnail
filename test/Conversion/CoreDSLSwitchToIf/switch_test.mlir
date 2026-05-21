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
// CHECK: %2 = hwarith.constant 3 : ui5
// CHECK: %3 = hwarith.constant 5 : ui3
// CHECK: %4 = hwarith.constant 2 : ui5
// CHECK: %5 = hwarith.constant 10 : ui4
// CHECK: %6 = hwarith.constant 1 : ui5
// CHECK: %7 = hwarith.constant 0 : ui5
// CHECK: %8 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
// CHECK: %9 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK: %10 = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK: %11 = coredsl.cast %10 : ui5 to ui32
// CHECK: %12 = coredsl.get @MEM[%11 : ui32] : ui8
// CHECK: %13 = coredsl.cast %12 : ui8 to ui32
// CHECK: %14 = coredsl.get @MEM[1:0] : ui16
// CHECK: %15 = coredsl.get @MEM[2] : ui8
// CHECK: %16 = hwarith.icmp eq %8, %7 : ui5, ui5
// CHECK: %17:3 = scf.if %16 -> (ui32, ui16, ui8) {
// CHECK:   scf.yield %13, %14, %15 : ui32, ui16, ui8
// CHECK: } else {
// CHECK:   %18 = hwarith.icmp eq %8, %6 : ui5, ui5
// CHECK:   %19:3 = scf.if %18 -> (ui32, ui16, ui8) {
// CHECK:     %20 = coredsl.cast %5 : ui4 to ui32
// CHECK:     scf.yield %20, %14, %15 : ui32, ui16, ui8
// CHECK:   } else {
// CHECK:     %20 = hwarith.icmp eq %8, %4 : ui5, ui5
// CHECK:     %21:3 = scf.if %20 -> (ui32, ui16, ui8) {
// CHECK:       %22 = coredsl.cast %3 : ui3 to ui16
// CHECK:       scf.yield %13, %22, %15 : ui32, ui16, ui8
// CHECK:     } else {
// CHECK:       %22 = hwarith.icmp eq %8, %2 : ui5, ui5
// CHECK:       %23:3 = scf.if %22 -> (ui32, ui16, ui8) {
// CHECK:         %24 = coredsl.cast %1 : ui3 to ui8
// CHECK:         scf.yield %13, %14, %24 : ui32, ui16, ui8
// CHECK:       } else {
// CHECK:         %24 = coredsl.cast %0 : ui1 to ui32
// CHECK:         %25 = coredsl.cast %0 : ui1 to ui16
// CHECK:         %26 = coredsl.cast %0 : ui1 to ui8
// CHECK:         scf.yield %24, %25, %26 : ui32, ui16, ui8
// CHECK:       }
// CHECK:       scf.yield %23#0, %23#1, %23#2 : ui32, ui16, ui8
// CHECK:     }
// CHECK:     scf.yield %21#0, %21#1, %21#2 : ui32, ui16, ui8
// CHECK:   }
// CHECK:   scf.yield %19#0, %19#1, %19#2 : ui32, ui16, ui8
// CHECK: }
// CHECK: coredsl.set @X[%9 : ui5] = %17#0 : ui32
// CHECK: coredsl.set @MEM[1:0] = %17#1 : ui16
// CHECK: coredsl.set @MEM[2] = %17#2 : ui8

    %rs2 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %1 = coredsl.cast %rd : ui5 to ui32
    %0 = coredsl.get @MEM[%1 : ui32] : ui8
    %2 = coredsl.cast %0 : ui8 to ui32
    %3 = coredsl.get @MEM[1:0] : ui16
    %4 = coredsl.get @MEM[2] : ui8
    %5 = hwarith.cast %rs2 : (ui5) -> i6
    %6, %7, %8 = scf.execute_region -> (ui32, ui16, ui8) {
      cf.switch %5 : i6, [
        default: ^default,
        0: ^case_0,
        1: ^case_1,
        2: ^case_2,
        3: ^case_3
      ]
      ^case_0():
        cf.br ^switch_end(%2, %3, %4 : ui32, ui16, ui8)
      ^case_1():
        %6 = hwarith.constant 10 : ui4
        %7 = coredsl.cast %6 : ui4 to ui32
        cf.br ^switch_end(%7, %3, %4 : ui32, ui16, ui8)
      ^case_2():
        %8 = hwarith.constant 5 : ui3
        %9 = coredsl.cast %8 : ui3 to ui16
        cf.br ^switch_end(%2, %9, %4 : ui32, ui16, ui8)
      ^case_3():
        %10 = hwarith.constant 7 : ui3
        %11 = coredsl.cast %10 : ui3 to ui8
        cf.br ^switch_end(%2, %3, %11 : ui32, ui16, ui8)
      ^default():
        %12 = hwarith.constant 1 : ui1
        %13 = coredsl.cast %12 : ui1 to ui32
        %14 = hwarith.constant 1 : ui1
        %15 = coredsl.cast %14 : ui1 to ui16
        %16 = hwarith.constant 1 : ui1
        %17 = coredsl.cast %16 : ui1 to ui8
        cf.br ^switch_end(%13, %15, %17 : ui32, ui16, ui8)
      ^switch_end(%18: ui32, %19: ui16, %20: ui8):
        scf.yield %18, %19, %20 : ui32, ui16, ui8
    }
    coredsl.set @X[%rs1 : ui5] = %6 : ui32
    coredsl.set @MEM[1:0] = %7 : ui16
    coredsl.set @MEM[2] = %8 : ui8
    coredsl.end
  }
  coredsl.instruction @NestedSwitch {lil.enc_immediates = [[["%TREENAIL_WAS_HERE_rs2_4_0", 4, 0, 0, "rs2"]], [["%TREENAIL_WAS_HERE_rs1_4_0", 4, 0, 0, "rs1"]], [["%TREENAIL_WAS_HERE_rd_4_0", 4, 0, 0, "rd"]]]} ("0000000", %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "001", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0101011") {

// CHECK: %0 = hwarith.constant 7 : ui3
// CHECK: %1 = hwarith.constant 3 : ui5
// CHECK: %2 = hwarith.constant 5 : ui3
// CHECK: %3 = hwarith.constant 10 : ui4
// CHECK: %4 = hwarith.constant 1 : ui5
// CHECK: %5 = hwarith.constant 2 : ui2
// CHECK: %6 = hwarith.constant 2 : ui5
// CHECK: %7 = hwarith.constant 1 : ui1
// CHECK: %8 = hwarith.constant 16 : ui5
// CHECK: %9 = hwarith.constant 0 : ui5
// CHECK: %10 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
// CHECK: %11 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK: %12 = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK: %13 = coredsl.cast %12 : ui5 to ui32
// CHECK: %14 = coredsl.get @MEM[%13 : ui32] : ui8
// CHECK: %15 = coredsl.cast %14 : ui8 to ui32
// CHECK: %16 = coredsl.get @MEM[1:0] : ui16
// CHECK: %17 = coredsl.get @MEM[2] : ui8
// CHECK: %18 = hwarith.icmp eq %10, %9 : ui5, ui5
// CHECK: %19:3 = scf.if %18 -> (ui16, ui32, ui8) {
// CHECK:   %20 = hwarith.icmp eq %12, %8 : ui5, ui5
// CHECK:   %21 = scf.if %20 -> (ui16) {
// CHECK:     %22 = hwarith.sub %16, %7 : (ui16, ui1) -> si17
// CHECK:     %23 = coredsl.cast %22 : si17 to ui16
// CHECK:     scf.yield %23 : ui16
// CHECK:   } else {
// CHECK:     %22 = hwarith.icmp eq %12, %6 : ui5, ui5
// CHECK:     %23 = scf.if %22 -> (ui16) {
// CHECK:       %24 = hwarith.add %16, %7 : (ui16, ui1) -> ui17
// CHECK:       %25 = coredsl.cast %24 : ui17 to ui16
// CHECK:       scf.yield %25 : ui16
// CHECK:     } else {
// CHECK:       %24 = hwarith.add %16, %5 : (ui16, ui2) -> ui17
// CHECK:       %25 = coredsl.cast %24 : ui17 to ui16
// CHECK:       scf.yield %25 : ui16
// CHECK:     }
// CHECK:     scf.yield %23 : ui16
// CHECK:   }
// CHECK:   scf.yield %21, %15, %17 : ui16, ui32, ui8
// CHECK: } else {
// CHECK:   %20 = hwarith.icmp eq %10, %4 : ui5, ui5
// CHECK:   %21:3 = scf.if %20 -> (ui16, ui32, ui8) {
// CHECK:     %22 = coredsl.cast %3 : ui4 to ui32
// CHECK:     scf.yield %16, %22, %17 : ui16, ui32, ui8
// CHECK:   } else {
// CHECK:     %22 = hwarith.icmp eq %10, %6 : ui5, ui5
// CHECK:     %23:2 = scf.if %22 -> (ui16, ui8) {
// CHECK:       %24 = coredsl.cast %2 : ui3 to ui16
// CHECK:       scf.yield %24, %17 : ui16, ui8
// CHECK:     } else {
// CHECK:       %24 = hwarith.icmp eq %10, %1 : ui5, ui5
// CHECK:       %25 = scf.if %24 -> (ui8) {
// CHECK:         %26 = coredsl.cast %0 : ui3 to ui8
// CHECK:         scf.yield %26 : ui8
// CHECK:       } else {
// CHECK:         scf.yield %17 : ui8
// CHECK:       }
// CHECK:       scf.yield %16, %25 : ui16, ui8
// CHECK:     }
// CHECK:     scf.yield %23#0, %15, %23#1 : ui16, ui32, ui8
// CHECK:   }
// CHECK:   scf.yield %21#0, %21#1, %21#2 : ui16, ui32, ui8
// CHECK: }
// CHECK: coredsl.set @X[%11 : ui5] = %19#1 : ui32
// CHECK: coredsl.set @MEM[1:0] = %19#0 : ui16
// CHECK: coredsl.set @MEM[2] = %19#2 : ui8

    %rs2 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %1 = coredsl.cast %rd : ui5 to ui32
    %0 = coredsl.get @MEM[%1 : ui32] : ui8
    %2 = coredsl.cast %0 : ui8 to ui32
    %3 = coredsl.get @MEM[1:0] : ui16
    %4 = coredsl.get @MEM[2] : ui8
    %5 = hwarith.cast %rs2 : (ui5) -> i6
    %6, %7, %8 = scf.execute_region -> (ui16, ui32, ui8) {
      cf.switch %5 : i6, [
        default: ^switch_end(%3, %2, %4 : ui16, ui32, ui8),
        0: ^case_0,
        1: ^case_1,
        2: ^case_2,
        3: ^case_3
      ]
      ^case_0():
        %6 = hwarith.cast %rd : (ui5) -> i6
        %7 = scf.execute_region -> (ui16) {
          cf.switch %6 : i6, [
            default: ^default,
            16: ^case_16,
            2: ^case_2
          ]
          ^case_16():
            %7 = hwarith.constant 1 : ui1
            %8 = hwarith.sub %3, %7 : (ui16, ui1) -> si17
            %9 = coredsl.cast %8 : si17 to ui16
            cf.br ^switch_end(%9 : ui16)
          ^case_2():
            %10 = hwarith.constant 1 : ui1
            %11 = hwarith.add %3, %10 : (ui16, ui1) -> ui17
            %12 = coredsl.cast %11 : ui17 to ui16
            cf.br ^switch_end(%12 : ui16)
          ^default():
            %13 = hwarith.constant 2 : ui2
            %14 = hwarith.add %3, %13 : (ui16, ui2) -> ui17
            %15 = coredsl.cast %14 : ui17 to ui16
            cf.br ^switch_end(%15 : ui16)
          ^switch_end(%16: ui16):
            scf.yield %16 : ui16
        }
        cf.br ^switch_end(%7, %2, %4 : ui16, ui32, ui8)
      ^case_1():
        %8 = hwarith.constant 10 : ui4
        %9 = coredsl.cast %8 : ui4 to ui32
        cf.br ^switch_end(%3, %9, %4 : ui16, ui32, ui8)
      ^case_2():
        %10 = hwarith.constant 5 : ui3
        %11 = coredsl.cast %10 : ui3 to ui16
        cf.br ^switch_end(%11, %2, %4 : ui16, ui32, ui8)
      ^case_3():
        %12 = hwarith.constant 7 : ui3
        %13 = coredsl.cast %12 : ui3 to ui8
        cf.br ^switch_end(%3, %2, %13 : ui16, ui32, ui8)
      ^switch_end(%14: ui16, %15: ui32, %16: ui8):
        scf.yield %14, %15, %16 : ui16, ui32, ui8
    }
    coredsl.set @X[%rs1 : ui5] = %7 : ui32
    coredsl.set @MEM[1:0] = %6 : ui16
    coredsl.set @MEM[2] = %8 : ui8
    coredsl.end
  }
}

