// RUN: shortnail-opt %s -merge-multiple-isaxes | shortnail-opt | FileCheck %s

// CHECK: coredsl.isax "merged" {
// CHECK:   coredsl.register core_x @MERGEDX[32] : ui32
// CHECK:   coredsl.register core_pc @MERGEDPC : ui32
// CHECK:   coredsl.addrspace core_mem @MERGEDMEM : (ui32) -> ui8
// CHECK:   coredsl.register local @MERGED0ADDR : ui32
// CHECK:   coredsl.register local @MERGED1ADDR = 0 : ui1
// CHECK:   coredsl.alias @MERGED0RIP_MARKUS = @MERGED0ADDR
// CHECK:   coredsl.alias @MERGED0DIE_HARD = @MERGEDX[5]
// CHECK:   coredsl.alias @MERGED1RIP_MARKUS = @MERGEDPC
// CHECK:   coredsl.alias @MERGED1DIE_HARD = @MERGED1ADDR
// CHECK:   coredsl.instruction @set_addr("000000000000", %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "000", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0001011"){
// CHECK:     %[[_VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[_VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[_VAR_2:.*]] = coredsl.get @MERGEDX[%[[_VAR_0]] : ui5] : ui32
// CHECK:     coredsl.set @MERGED0RIP_MARKUS = %[[_VAR_2]] : ui32
// CHECK:     coredsl.set @MERGED0DIE_HARD = %[[_VAR_2]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @lw_inc(%TREENAIL_WAS_HERE_simm_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0101011"){
// CHECK:     %[[_VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_simm_11_0 : ui12 to ui12
// CHECK:     %[[_VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[_VAR_2:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[_VAR_3:.*]] = coredsl.get @MERGED0ADDR : ui32
// CHECK:     %[[_VAR_4:.*]] = coredsl.get @MERGEDMEM[%[[_VAR_3]] : ui32, 3:0] : ui32
// CHECK:     %[[_VAR_5:.*]] = coredsl.cast %[[_VAR_0]] : ui12 to si12
// CHECK:     %[[_VAR_6:.*]] = hwarith.add %[[_VAR_5]], %[[_VAR_3]] : (si12, ui32) -> si34
// CHECK:     %[[_VAR_7:.*]] = coredsl.cast %[[_VAR_6]] : si34 to ui32
// CHECK:     coredsl.set @MERGED0ADDR = %[[_VAR_7]] : ui32
// CHECK:     coredsl.set @MERGEDX[%[[_VAR_2]] : ui5] = %[[_VAR_4]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @sw_inc(%TREENAIL_WAS_HERE_simm_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1011011"){
// CHECK:     %[[_VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_simm_11_0 : ui12 to ui12
// CHECK:     %[[_VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[_VAR_2:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[_VAR_3:.*]] = coredsl.get @MERGED0ADDR : ui32
// CHECK:     %[[_VAR_4:.*]] = coredsl.get @MERGEDX[%[[_VAR_1]] : ui5] : ui32
// CHECK:     coredsl.set @MERGEDMEM[%[[_VAR_3]] : ui32, 3:0] = %[[_VAR_4]] : ui32
// CHECK:     %[[_VAR_5:.*]] = coredsl.cast %[[_VAR_0]] : ui12 to si12
// CHECK:     %[[_VAR_6:.*]] = hwarith.add %[[_VAR_5]], %[[_VAR_3]] : (si12, ui32) -> si34
// CHECK:     %[[_VAR_7:.*]] = coredsl.cast %[[_VAR_6]] : si34 to ui32
// CHECK:     coredsl.set @MERGED0ADDR = %[[_VAR_7]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @cv_beqimm(%TREENAIL_WAS_HERE_imm12_11_11 : ui1, %TREENAIL_WAS_HERE_imm12_9_4 : ui6, %TREENAIL_WAS_HERE_imm5_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "101", %TREENAIL_WAS_HERE_imm12_3_0 : ui4, %TREENAIL_WAS_HERE_imm12_10_10 : ui1, "0001011"){
// CHECK:     %[[_VAR_0:.*]] = coredsl.concat %TREENAIL_WAS_HERE_imm12_9_4, %TREENAIL_WAS_HERE_imm12_3_0 : ui6, ui4
// CHECK:     %[[_VAR_1:.*]] = coredsl.concat %TREENAIL_WAS_HERE_imm12_10_10, %[[_VAR_0]] : ui1, ui10
// CHECK:     %[[_VAR_2:.*]] = coredsl.concat %TREENAIL_WAS_HERE_imm12_11_11, %[[_VAR_1]] : ui1, ui11
// CHECK:     %[[_VAR_3:.*]] = coredsl.cast %[[_VAR_2]] : ui12 to ui12
// CHECK:     %[[_VAR_4:.*]] = coredsl.cast %TREENAIL_WAS_HERE_imm5_4_0 : ui5 to ui5
// CHECK:     %[[_VAR_5:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[_VAR_6:.*]] = hwarith.constant 0 : ui1
// CHECK:     %[[_VAR_7:.*]] = coredsl.concat %[[_VAR_3]], %[[_VAR_6]] : ui12, ui1
// CHECK:     %[[_VAR_8:.*]] = coredsl.cast %[[_VAR_7]] : ui13 to si13
// CHECK:     %[[_VAR_9:.*]] = coredsl.get @MERGEDX[%[[_VAR_5]] : ui5] : ui32
// CHECK:     %[[_VAR_10:.*]] = coredsl.cast %[[_VAR_4]] : ui5 to si5
// CHECK:     %[[_VAR_11:.*]] = hwarith.icmp eq %[[_VAR_9]], %[[_VAR_10]] : ui32, si5
// CHECK:     scf.if %[[_VAR_11]] {
// CHECK:       %[[_VAR_13:.*]] = coredsl.get @MERGEDPC : ui32
// CHECK:       %[[_VAR_14:.*]] = hwarith.add %[[_VAR_13]], %[[_VAR_8]] : (ui32, si13) -> si34
// CHECK:       %[[_VAR_15:.*]] = coredsl.cast %[[_VAR_14]] : si34 to ui32
// CHECK:       coredsl.set @MERGED1RIP_MARKUS = %[[_VAR_15]] : ui32
// CHECK:     }
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.always @HARDCORE {
// CHECK:     %[[_VAR_0:.*]] = coredsl.get @MERGED1DIE_HARD : ui1
// CHECK:     %[[_VAR_1:.*]] = hwarith.constant 1 : ui1
// CHECK:     %[[_VAR_2:.*]] = coredsl.xor %[[_VAR_0]], %[[_VAR_1]] : ui1, ui1
// CHECK:     coredsl.set @MERGED1ADDR = %[[_VAR_2]] : ui1
// CHECK:     %[[_VAR_3:.*]] = coredsl.cast %[[_VAR_0]] : ui1 to i1
// CHECK:     scf.if %[[_VAR_3]] {
// CHECK:       %[[_VAR_4:.*]] = coredsl.get @MERGEDPC : ui32
// CHECK:       %[[_VAR_5:.*]] = hwarith.constant 4 : ui3
// CHECK:       %[[_VAR_6:.*]] = hwarith.add %[[_VAR_4]], %[[_VAR_5]] : (ui32, ui3) -> ui33
// CHECK:       %[[_VAR_7:.*]] = coredsl.cast %[[_VAR_6]] : ui33 to ui32
// CHECK:       coredsl.set @MERGEDPC = %[[_VAR_7]] : ui32
// CHECK:     }
// CHECK:     coredsl.end
// CHECK:   }
// CHECK: }

coredsl.isax "autoinc" {
  coredsl.register core_x @X[32] : ui32
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8
  coredsl.register local @ADDR : ui32
  coredsl.alias @RIP_MARKUS = @ADDR
  coredsl.alias @DIE_HARD = @X[5]
  coredsl.instruction @set_addr("000000000000", %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "000", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0001011") {
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    coredsl.set @RIP_MARKUS = %0 : ui32
    coredsl.set @DIE_HARD = %0 : ui32
    coredsl.end
  }
  coredsl.instruction @lw_inc(%TREENAIL_WAS_HERE_simm_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0101011") {
    %simm = coredsl.cast %TREENAIL_WAS_HERE_simm_11_0 : ui12 to ui12
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @ADDR : ui32
    %1 = coredsl.get @MEM[%0 : ui32, 3:0] : ui32
    %2 = coredsl.cast %simm : ui12 to si12
    %3 = hwarith.add %2, %0 : (si12, ui32) -> si34
    %4 = coredsl.cast %3 : si34 to ui32
    coredsl.set @ADDR = %4 : ui32
    coredsl.set @X[%rd : ui5] = %1 : ui32
    coredsl.end
  }
  coredsl.instruction @sw_inc(%TREENAIL_WAS_HERE_simm_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1011011") {
    %simm = coredsl.cast %TREENAIL_WAS_HERE_simm_11_0 : ui12 to ui12
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @ADDR : ui32
    %1 = coredsl.get @X[%rs1 : ui5] : ui32
    coredsl.set @MEM[%0 : ui32, 3:0] = %1 : ui32
    %2 = coredsl.cast %simm : ui12 to si12
    %3 = hwarith.add %2, %0 : (si12, ui32) -> si34
    %4 = coredsl.cast %3 : si34 to ui32
    coredsl.set @ADDR = %4 : ui32
    coredsl.end
  }
}
coredsl.isax "brimm" {
  coredsl.register core_x @X[32] : ui32
  coredsl.register core_pc @PC : ui32
  coredsl.register local @ADDR = 0 : ui1
  coredsl.alias @RIP_MARKUS = @PC
  coredsl.alias @DIE_HARD = @ADDR
  coredsl.instruction @cv_beqimm(%TREENAIL_WAS_HERE_imm12_11_11 : ui1, %TREENAIL_WAS_HERE_imm12_9_4 : ui6, %TREENAIL_WAS_HERE_imm5_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "101", %TREENAIL_WAS_HERE_imm12_3_0 : ui4, %TREENAIL_WAS_HERE_imm12_10_10 : ui1, "0001011") {
    %TREENAIL_WAS_HERE_0 = coredsl.concat %TREENAIL_WAS_HERE_imm12_9_4, %TREENAIL_WAS_HERE_imm12_3_0 : ui6, ui4
    %TREENAIL_WAS_HERE_1 = coredsl.concat %TREENAIL_WAS_HERE_imm12_10_10, %TREENAIL_WAS_HERE_0 : ui1, ui10
    %TREENAIL_WAS_HERE_2 = coredsl.concat %TREENAIL_WAS_HERE_imm12_11_11, %TREENAIL_WAS_HERE_1 : ui1, ui11
    %imm12 = coredsl.cast %TREENAIL_WAS_HERE_2 : ui12 to ui12
    %imm5 = coredsl.cast %TREENAIL_WAS_HERE_imm5_4_0 : ui5 to ui5
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %0 = hwarith.constant 0 : ui1
    %1 = coredsl.concat %imm12, %0 : ui12, ui1
    %2 = coredsl.cast %1 : ui13 to si13
    %3 = coredsl.get @X[%rs1 : ui5] : ui32
    %4 = coredsl.cast %imm5 : ui5 to si5
    %6 = hwarith.icmp eq %3, %4 : ui32, si5
    scf.if %6 {
      %7 = coredsl.get @PC : ui32
      %8 = hwarith.add %7, %2 : (ui32, si13) -> si34
      %9 = coredsl.cast %8 : si34 to ui32
      coredsl.set @RIP_MARKUS = %9 : ui32
    }
    coredsl.end
  }

  coredsl.always @HARDCORE {
    %cond = coredsl.get @DIE_HARD : ui1
    %1 = hwarith.constant 1 : ui1
    %newToggle = coredsl.xor %cond, %1 : ui1, ui1
    coredsl.set @ADDR = %newToggle : ui1
    %cc = coredsl.cast %cond : ui1 to i1
    scf.if %cc {
      %pc = coredsl.get @PC : ui32
      %update = hwarith.constant 4 : ui3
      %tmp = hwarith.add %pc, %update : (ui32, ui3) -> ui33
      %newPc = coredsl.cast %tmp : ui33 to ui32
      coredsl.set @PC = %newPc : ui32
    }
    coredsl.end
  }
}

