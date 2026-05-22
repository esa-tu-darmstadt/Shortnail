// RUN: shortnail-opt %s -coredsl-switch-to-if -canonicalize | shortnail-opt | FileCheck %s

coredsl.isax "SwitchStmt" {
  coredsl.register core_x @X[32] : ui32
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8
  coredsl.instruction @Simple {lil.enc_immediates = [[["%TREENAIL_WAS_HERE_rs2_4_0", 4, 0, 0, "rs2"]], [["%TREENAIL_WAS_HERE_rs1_4_0", 4, 0, 0, "rs1"]], [["%TREENAIL_WAS_HERE_rd_4_0", 4, 0, 0, "rd"]]]} ("0000000", %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "001", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0101011") {
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
  coredsl.instruction @DefaultOnlySwitch {lil.enc_immediates = [[["%TREENAIL_WAS_HERE_rs2_4_0", 4, 0, 0, "rs2"]], [["%TREENAIL_WAS_HERE_rs1_4_0", 4, 0, 0, "rs1"]], [["%TREENAIL_WAS_HERE_rd_4_0", 4, 0, 0, "rd"]]]} ("0000000", %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "001", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0101011") {
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
        default: ^default
      ]
      ^default():
        %7 = coredsl.cast %rs1 : ui5 to ui32
        %6 = coredsl.get @MEM[%7 : ui32] : ui8
        %8 = coredsl.cast %6 : ui8 to ui32
        %9 = hwarith.constant 10 : ui4
        %10 = coredsl.cast %9 : ui4 to ui16
        %11 = hwarith.constant 5 : ui3
        %12 = coredsl.cast %11 : ui3 to ui8
        cf.br ^switch_end(%8, %10, %12 : ui32, ui16, ui8)
      ^switch_end(%13: ui32, %14: ui16, %15: ui8):
        scf.yield %13, %14, %15 : ui32, ui16, ui8
    }
    coredsl.set @X[%rs1 : ui5] = %6 : ui32
    coredsl.set @MEM[1:0] = %7 : ui16
    coredsl.set @MEM[2] = %8 : ui8
    coredsl.end
  }
}

// CHECK-LABEL:   coredsl.isax "SwitchStmt" {
// CHECK:           coredsl.register core_x @X[32]  : ui32
// CHECK:           coredsl.addrspace core_mem @MEM : (ui32) -> ui8
// CHECK:           coredsl.instruction @Simple {lil.enc_immediates = {{\[\[}}["%[[VAL_0:.*]]", 4, 0, 0, "rs2"]], {{\[\[}}"%[[VAL_1:.*]]", 4, 0, 0, "rs1"]], {{\[\[}}"%[[VAL_2:.*]]", 4, 0, 0, "rd"]]]}("0000000", %[[VAL_0]] : ui5, %[[VAL_1]] : ui5, "001", %[[VAL_2]] : ui5, "0101011"){
// CHECK:             %[[CONSTANT_0:.*]] = hwarith.constant 1 : ui1
// CHECK:             %[[CONSTANT_1:.*]] = hwarith.constant 7 : ui3
// CHECK:             %[[CONSTANT_2:.*]] = hwarith.constant 3 : ui5
// CHECK:             %[[CONSTANT_3:.*]] = hwarith.constant 5 : ui3
// CHECK:             %[[CONSTANT_4:.*]] = hwarith.constant 2 : ui5
// CHECK:             %[[CONSTANT_5:.*]] = hwarith.constant 10 : ui4
// CHECK:             %[[CONSTANT_6:.*]] = hwarith.constant 1 : ui5
// CHECK:             %[[CONSTANT_7:.*]] = hwarith.constant 0 : ui5
// CHECK:             %[[CAST_0:.*]] = coredsl.cast %[[VAL_0]] : ui5 to ui5
// CHECK:             %[[CAST_1:.*]] = coredsl.cast %[[VAL_1]] : ui5 to ui5
// CHECK:             %[[CAST_2:.*]] = coredsl.cast %[[VAL_2]] : ui5 to ui5
// CHECK:             %[[CAST_3:.*]] = coredsl.cast %[[CAST_2]] : ui5 to ui32
// CHECK:             %[[GET_0:.*]] = coredsl.get @MEM{{\[}}%[[CAST_3]] : ui32] : ui8
// CHECK:             %[[CAST_4:.*]] = coredsl.cast %[[GET_0]] : ui8 to ui32
// CHECK:             %[[GET_1:.*]] = coredsl.get @MEM[1:0] : ui16
// CHECK:             %[[GET_2:.*]] = coredsl.get @MEM[2] : ui8
// CHECK:             %[[ICMP_0:.*]] = hwarith.icmp eq %[[CAST_0]], %[[CONSTANT_7]] : ui5, ui5
// CHECK:             %[[IF_0:.*]]:3 = scf.if %[[ICMP_0]] -> (ui32, ui16, ui8) {
// CHECK:               scf.yield %[[CAST_4]], %[[GET_1]], %[[GET_2]] : ui32, ui16, ui8
// CHECK:             } else {
// CHECK:               %[[ICMP_1:.*]] = hwarith.icmp eq %[[CAST_0]], %[[CONSTANT_6]] : ui5, ui5
// CHECK:               %[[IF_1:.*]]:3 = scf.if %[[ICMP_1]] -> (ui32, ui16, ui8) {
// CHECK:                 %[[CAST_5:.*]] = coredsl.cast %[[CONSTANT_5]] : ui4 to ui32
// CHECK:                 scf.yield %[[CAST_5]], %[[GET_1]], %[[GET_2]] : ui32, ui16, ui8
// CHECK:               } else {
// CHECK:                 %[[ICMP_2:.*]] = hwarith.icmp eq %[[CAST_0]], %[[CONSTANT_4]] : ui5, ui5
// CHECK:                 %[[IF_2:.*]]:3 = scf.if %[[ICMP_2]] -> (ui32, ui16, ui8) {
// CHECK:                   %[[CAST_6:.*]] = coredsl.cast %[[CONSTANT_3]] : ui3 to ui16
// CHECK:                   scf.yield %[[CAST_4]], %[[CAST_6]], %[[GET_2]] : ui32, ui16, ui8
// CHECK:                 } else {
// CHECK:                   %[[ICMP_3:.*]] = hwarith.icmp eq %[[CAST_0]], %[[CONSTANT_2]] : ui5, ui5
// CHECK:                   %[[IF_3:.*]]:3 = scf.if %[[ICMP_3]] -> (ui32, ui16, ui8) {
// CHECK:                     %[[CAST_7:.*]] = coredsl.cast %[[CONSTANT_1]] : ui3 to ui8
// CHECK:                     scf.yield %[[CAST_4]], %[[GET_1]], %[[CAST_7]] : ui32, ui16, ui8
// CHECK:                   } else {
// CHECK:                     %[[CAST_8:.*]] = coredsl.cast %[[CONSTANT_0]] : ui1 to ui32
// CHECK:                     %[[CAST_9:.*]] = coredsl.cast %[[CONSTANT_0]] : ui1 to ui16
// CHECK:                     %[[CAST_10:.*]] = coredsl.cast %[[CONSTANT_0]] : ui1 to ui8
// CHECK:                     scf.yield %[[CAST_8]], %[[CAST_9]], %[[CAST_10]] : ui32, ui16, ui8
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_3:.*]]#0, %[[VAL_3]]#1, %[[VAL_3]]#2 : ui32, ui16, ui8
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_4:.*]]#0, %[[VAL_4]]#1, %[[VAL_4]]#2 : ui32, ui16, ui8
// CHECK:               }
// CHECK:               scf.yield %[[VAL_5:.*]]#0, %[[VAL_5]]#1, %[[VAL_5]]#2 : ui32, ui16, ui8
// CHECK:             }
// CHECK:             coredsl.set @X{{\[}}%[[CAST_1]] : ui5] = %[[VAL_6:.*]]#0 : ui32
// CHECK:             coredsl.set @MEM[1:0] = %[[VAL_6]]#1 : ui16
// CHECK:             coredsl.set @MEM[2] = %[[VAL_6]]#2 : ui8
// CHECK:             coredsl.end
// CHECK:           }
// CHECK:           coredsl.instruction @NestedSwitch {lil.enc_immediates = {{\[\[}}["%[[VAL_7:.*]]", 4, 0, 0, "rs2"]], {{\[\[}}"%[[VAL_8:.*]]", 4, 0, 0, "rs1"]], {{\[\[}}"%[[VAL_9:.*]]", 4, 0, 0, "rd"]]]}("0000000", %[[VAL_7]] : ui5, %[[VAL_8]] : ui5, "001", %[[VAL_9]] : ui5, "0101011"){
// CHECK:             %[[CONSTANT_8:.*]] = hwarith.constant 7 : ui3
// CHECK:             %[[CONSTANT_9:.*]] = hwarith.constant 3 : ui5
// CHECK:             %[[CONSTANT_10:.*]] = hwarith.constant 5 : ui3
// CHECK:             %[[CONSTANT_11:.*]] = hwarith.constant 10 : ui4
// CHECK:             %[[CONSTANT_12:.*]] = hwarith.constant 1 : ui5
// CHECK:             %[[CONSTANT_13:.*]] = hwarith.constant 2 : ui2
// CHECK:             %[[CONSTANT_14:.*]] = hwarith.constant 2 : ui5
// CHECK:             %[[CONSTANT_15:.*]] = hwarith.constant 1 : ui1
// CHECK:             %[[CONSTANT_16:.*]] = hwarith.constant 16 : ui5
// CHECK:             %[[CONSTANT_17:.*]] = hwarith.constant 0 : ui5
// CHECK:             %[[CAST_11:.*]] = coredsl.cast %[[VAL_7]] : ui5 to ui5
// CHECK:             %[[CAST_12:.*]] = coredsl.cast %[[VAL_8]] : ui5 to ui5
// CHECK:             %[[CAST_13:.*]] = coredsl.cast %[[VAL_9]] : ui5 to ui5
// CHECK:             %[[CAST_14:.*]] = coredsl.cast %[[CAST_13]] : ui5 to ui32
// CHECK:             %[[GET_3:.*]] = coredsl.get @MEM{{\[}}%[[CAST_14]] : ui32] : ui8
// CHECK:             %[[CAST_15:.*]] = coredsl.cast %[[GET_3]] : ui8 to ui32
// CHECK:             %[[GET_4:.*]] = coredsl.get @MEM[1:0] : ui16
// CHECK:             %[[GET_5:.*]] = coredsl.get @MEM[2] : ui8
// CHECK:             %[[ICMP_4:.*]] = hwarith.icmp eq %[[CAST_11]], %[[CONSTANT_17]] : ui5, ui5
// CHECK:             %[[IF_4:.*]]:3 = scf.if %[[ICMP_4]] -> (ui16, ui32, ui8) {
// CHECK:               %[[ICMP_5:.*]] = hwarith.icmp eq %[[CAST_13]], %[[CONSTANT_16]] : ui5, ui5
// CHECK:               %[[IF_5:.*]] = scf.if %[[ICMP_5]] -> (ui16) {
// CHECK:                 %[[SUB_0:.*]] = hwarith.sub %[[GET_4]], %[[CONSTANT_15]] : (ui16, ui1) -> si17
// CHECK:                 %[[CAST_16:.*]] = coredsl.cast %[[SUB_0]] : si17 to ui16
// CHECK:                 scf.yield %[[CAST_16]] : ui16
// CHECK:               } else {
// CHECK:                 %[[ICMP_6:.*]] = hwarith.icmp eq %[[CAST_13]], %[[CONSTANT_14]] : ui5, ui5
// CHECK:                 %[[IF_6:.*]] = scf.if %[[ICMP_6]] -> (ui16) {
// CHECK:                   %[[ADD_0:.*]] = hwarith.add %[[GET_4]], %[[CONSTANT_15]] : (ui16, ui1) -> ui17
// CHECK:                   %[[CAST_17:.*]] = coredsl.cast %[[ADD_0]] : ui17 to ui16
// CHECK:                   scf.yield %[[CAST_17]] : ui16
// CHECK:                 } else {
// CHECK:                   %[[ADD_1:.*]] = hwarith.add %[[GET_4]], %[[CONSTANT_13]] : (ui16, ui2) -> ui17
// CHECK:                   %[[CAST_18:.*]] = coredsl.cast %[[ADD_1]] : ui17 to ui16
// CHECK:                   scf.yield %[[CAST_18]] : ui16
// CHECK:                 }
// CHECK:                 scf.yield %[[IF_6]] : ui16
// CHECK:               }
// CHECK:               scf.yield %[[IF_5]], %[[CAST_15]], %[[GET_5]] : ui16, ui32, ui8
// CHECK:             } else {
// CHECK:               %[[ICMP_7:.*]] = hwarith.icmp eq %[[CAST_11]], %[[CONSTANT_12]] : ui5, ui5
// CHECK:               %[[IF_7:.*]]:3 = scf.if %[[ICMP_7]] -> (ui16, ui32, ui8) {
// CHECK:                 %[[CAST_19:.*]] = coredsl.cast %[[CONSTANT_11]] : ui4 to ui32
// CHECK:                 scf.yield %[[GET_4]], %[[CAST_19]], %[[GET_5]] : ui16, ui32, ui8
// CHECK:               } else {
// CHECK:                 %[[ICMP_8:.*]] = hwarith.icmp eq %[[CAST_11]], %[[CONSTANT_14]] : ui5, ui5
// CHECK:                 %[[IF_8:.*]]:2 = scf.if %[[ICMP_8]] -> (ui16, ui8) {
// CHECK:                   %[[CAST_20:.*]] = coredsl.cast %[[CONSTANT_10]] : ui3 to ui16
// CHECK:                   scf.yield %[[CAST_20]], %[[GET_5]] : ui16, ui8
// CHECK:                 } else {
// CHECK:                   %[[ICMP_9:.*]] = hwarith.icmp eq %[[CAST_11]], %[[CONSTANT_9]] : ui5, ui5
// CHECK:                   %[[IF_9:.*]] = scf.if %[[ICMP_9]] -> (ui8) {
// CHECK:                     %[[CAST_21:.*]] = coredsl.cast %[[CONSTANT_8]] : ui3 to ui8
// CHECK:                     scf.yield %[[CAST_21]] : ui8
// CHECK:                   } else {
// CHECK:                     scf.yield %[[GET_5]] : ui8
// CHECK:                   }
// CHECK:                   scf.yield %[[GET_4]], %[[IF_9]] : ui16, ui8
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_10:.*]]#0, %[[CAST_15]], %[[VAL_10]]#1 : ui16, ui32, ui8
// CHECK:               }
// CHECK:               scf.yield %[[VAL_11:.*]]#0, %[[VAL_11]]#1, %[[VAL_11]]#2 : ui16, ui32, ui8
// CHECK:             }
// CHECK:             coredsl.set @X{{\[}}%[[CAST_12]] : ui5] = %[[VAL_12:.*]]#1 : ui32
// CHECK:             coredsl.set @MEM[1:0] = %[[VAL_12]]#0 : ui16
// CHECK:             coredsl.set @MEM[2] = %[[VAL_12]]#2 : ui8
// CHECK:             coredsl.end
// CHECK:           }
// CHECK:           coredsl.instruction @DefaultOnlySwitch {lil.enc_immediates = {{\[\[}}["%[[VAL_13:.*]]", 4, 0, 0, "rs2"]], {{\[\[}}"%[[VAL_14:.*]]", 4, 0, 0, "rs1"]], {{\[\[}}"%[[VAL_15:.*]]", 4, 0, 0, "rd"]]]}("0000000", %[[VAL_13]] : ui5, %[[VAL_14]] : ui5, "001", %[[VAL_15]] : ui5, "0101011"){
// CHECK:             %[[CONSTANT_18:.*]] = hwarith.constant 5 : ui3
// CHECK:             %[[CONSTANT_19:.*]] = hwarith.constant 10 : ui4
// CHECK:             %[[CAST_22:.*]] = coredsl.cast %[[VAL_14]] : ui5 to ui5
// CHECK:             %[[CAST_23:.*]] = coredsl.cast %[[VAL_15]] : ui5 to ui5
// CHECK:             %[[CAST_24:.*]] = coredsl.cast %[[CAST_23]] : ui5 to ui32
// CHECK:             %[[GET_6:.*]] = coredsl.get @MEM{{\[}}%[[CAST_24]] : ui32] : ui8
// CHECK:             %[[GET_7:.*]] = coredsl.get @MEM[1:0] : ui16
// CHECK:             %[[GET_8:.*]] = coredsl.get @MEM[2] : ui8
// CHECK:             %[[CAST_25:.*]] = coredsl.cast %[[CAST_22]] : ui5 to ui32
// CHECK:             %[[GET_9:.*]] = coredsl.get @MEM{{\[}}%[[CAST_25]] : ui32] : ui8
// CHECK:             %[[CAST_26:.*]] = coredsl.cast %[[GET_9]] : ui8 to ui32
// CHECK:             %[[CAST_27:.*]] = coredsl.cast %[[CONSTANT_19]] : ui4 to ui16
// CHECK:             %[[CAST_28:.*]] = coredsl.cast %[[CONSTANT_18]] : ui3 to ui8
// CHECK:             coredsl.set @X{{\[}}%[[CAST_22]] : ui5] = %[[CAST_26]] : ui32
// CHECK:             coredsl.set @MEM[1:0] = %[[CAST_27]] : ui16
// CHECK:             coredsl.set @MEM[2] = %[[CAST_28]] : ui8
// CHECK:             coredsl.end
// CHECK:           }
// CHECK:         }
