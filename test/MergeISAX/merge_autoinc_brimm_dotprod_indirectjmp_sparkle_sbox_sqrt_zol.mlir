// RUN: shortnail-opt %s -merge-multiple-isaxes | shortnail-opt | FileCheck %s

// CHECK: coredsl.isax "merged" {
// CHECK:   coredsl.register core_x @MERGEDX[32] : ui32
// CHECK:   coredsl.register core_pc @MERGEDPC : ui32
// CHECK:   coredsl.addrspace core_mem @MERGEDMEM : (ui32) -> ui8
// CHECK:   coredsl.register local @MERGED0ADDR : ui32
// CHECK:   coredsl.register local const @MERGED4SBOX[256] = [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22] : ui8
// CHECK:   coredsl.register local const @MERGED5ROT_0[4] = [31, 17, 0, 24] : ui8
// CHECK:   coredsl.register local const @MERGED5ROT_1[4] = [24, 17, 31, 16] : ui8
// CHECK:   coredsl.register local const @MERGED5RCON[8] = [3084996962, 3211876480, 951376470, 844003128, 3138487787, 1333558103, 3485442504, 3266521405] : ui32
// CHECK:   coredsl.register local @MERGED7ZOLSTARTPC : ui32
// CHECK:   coredsl.register local @MERGED7ZOLENDPC : ui32
// CHECK:   coredsl.register local @MERGED7ZOLCOUNTER : ui32
// CHECK:   func.func @MERGED5ROTR(%arg0: ui32, %arg1: ui8) -> ui32 {
// CHECK:     %[[VAR_0:.*]] = coredsl.shift_right %arg0, %arg1 : ui32, ui8
// CHECK:     %[[VAR_1:.*]] = hwarith.constant 32 : ui6
// CHECK:     %[[VAR_2:.*]] = hwarith.sub %[[VAR_1]], %arg1 : (ui6, ui8) -> si9
// CHECK:     %[[VAR_3:.*]] = coredsl.shift_left %arg0, %[[VAR_2]] : ui32, si9
// CHECK:     %[[VAR_4:.*]] = coredsl.or %[[VAR_0]], %[[VAR_3]] : ui32, ui32
// CHECK:     return %[[VAR_4]] : ui32
// CHECK:   }
// CHECK:   func.func @MERGED5ELL(%arg0: ui32) -> ui32 {
// CHECK:     %[[VAR_0:.*]] = hwarith.constant 16 : ui5
// CHECK:     %[[VAR_1:.*]] = coredsl.shift_left %arg0, %[[VAR_0]] : ui32, ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.xor %arg0, %[[VAR_1]] : ui32, ui32
// CHECK:     %[[VAR_3:.*]] = hwarith.constant 16 : ui5
// CHECK:     %[[VAR_4:.*]] = coredsl.cast %[[VAR_3]] : ui5 to ui8
// CHECK:     %[[VAR_5:.*]] = call @MERGED5ROTR(%[[VAR_2]], %[[VAR_4]]) : (ui32, ui8) -> ui32
// CHECK:     return %[[VAR_5]] : ui32
// CHECK:   }
// CHECK:   coredsl.instruction @set_addr("000000000000", %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "000", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0001011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.get @MERGEDX[%[[VAR_0]] : ui5] : ui32
// CHECK:     coredsl.set @MERGED0ADDR = %[[VAR_2]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @lw_inc(%TREENAIL_WAS_HERE_simm_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0101011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_simm_11_0 : ui12 to ui12
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[VAR_3:.*]] = coredsl.get @MERGED0ADDR : ui32
// CHECK:     %[[VAR_4:.*]] = coredsl.get @MERGEDMEM[%[[VAR_3]] : ui32, 3:0] : ui32
// CHECK:     %[[VAR_5:.*]] = coredsl.cast %[[VAR_0]] : ui12 to si12
// CHECK:     %[[VAR_6:.*]] = hwarith.add %[[VAR_5]], %[[VAR_3]] : (si12, ui32) -> si34
// CHECK:     %[[VAR_7:.*]] = coredsl.cast %[[VAR_6]] : si34 to ui32
// CHECK:     coredsl.set @MERGED0ADDR = %[[VAR_7]] : ui32
// CHECK:     coredsl.set @MERGEDX[%[[VAR_2]] : ui5] = %[[VAR_4]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @sw_inc(%TREENAIL_WAS_HERE_simm_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1011011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_simm_11_0 : ui12 to ui12
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[VAR_3:.*]] = coredsl.get @MERGED0ADDR : ui32
// CHECK:     %[[VAR_4:.*]] = coredsl.get @MERGEDX[%[[VAR_1]] : ui5] : ui32
// CHECK:     coredsl.set @MERGEDMEM[%[[VAR_3]] : ui32, 3:0] = %[[VAR_4]] : ui32
// CHECK:     %[[VAR_5:.*]] = coredsl.cast %[[VAR_0]] : ui12 to si12
// CHECK:     %[[VAR_6:.*]] = hwarith.add %[[VAR_5]], %[[VAR_3]] : (si12, ui32) -> si34
// CHECK:     %[[VAR_7:.*]] = coredsl.cast %[[VAR_6]] : si34 to ui32
// CHECK:     coredsl.set @MERGED0ADDR = %[[VAR_7]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @cv_beqimm(%TREENAIL_WAS_HERE_imm12_11_11 : ui1, %TREENAIL_WAS_HERE_imm12_9_4 : ui6, %TREENAIL_WAS_HERE_imm5_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "101", %TREENAIL_WAS_HERE_imm12_3_0 : ui4, %TREENAIL_WAS_HERE_imm12_10_10 : ui1, "0001011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.concat %TREENAIL_WAS_HERE_imm12_9_4, %TREENAIL_WAS_HERE_imm12_3_0 : ui6, ui4
// CHECK:     %[[VAR_1:.*]] = coredsl.concat %TREENAIL_WAS_HERE_imm12_10_10, %[[VAR_0]] : ui1, ui10
// CHECK:     %[[VAR_2:.*]] = coredsl.concat %TREENAIL_WAS_HERE_imm12_11_11, %[[VAR_1]] : ui1, ui11
// CHECK:     %[[VAR_3:.*]] = coredsl.cast %[[VAR_2]] : ui12 to ui12
// CHECK:     %[[VAR_4:.*]] = coredsl.cast %TREENAIL_WAS_HERE_imm5_4_0 : ui5 to ui5
// CHECK:     %[[VAR_5:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[VAR_6:.*]] = hwarith.constant 0 : ui1
// CHECK:     %[[VAR_7:.*]] = coredsl.concat %[[VAR_3]], %[[VAR_6]] : ui12, ui1
// CHECK:     %[[VAR_8:.*]] = coredsl.cast %[[VAR_7]] : ui13 to si13
// CHECK:     %[[VAR_9:.*]] = coredsl.get @MERGEDX[%[[VAR_5]] : ui5] : ui32
// CHECK:     %[[VAR_10:.*]] = coredsl.cast %[[VAR_4]] : ui5 to si5
// CHECK:     %[[VAR_11:.*]] = hwarith.icmp eq %[[VAR_9]], %[[VAR_10]] : ui32, si5
// CHECK:     scf.if %[[VAR_11]] {
// CHECK:       %[[VAR_13:.*]] = coredsl.get @MERGEDPC : ui32
// CHECK:       %[[VAR_14:.*]] = hwarith.add %[[VAR_13]], %[[VAR_8]] : (ui32, si13) -> si34
// CHECK:       %[[VAR_15:.*]] = coredsl.cast %[[VAR_14]] : si34 to ui32
// CHECK:       coredsl.set @MERGEDPC = %[[VAR_15]] : ui32
// CHECK:     }
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @DOTP("0001001", %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "000", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0001011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[VAR_3:.*]] = hwarith.constant 0 : ui1
// CHECK:     %[[VAR_4:.*]] = coredsl.cast %[[VAR_3]] : ui1 to si32
// CHECK:     %[[CONST_0:.*]] = hw.constant 0 : i32
// CHECK:     %[[CONST_1:.*]] = hw.constant 32 : i32
// CHECK:     %[[CONST_2:.*]] = hw.constant 8 : i32
// CHECK:     %[[VAR_5:.*]] = scf.for %arg0 = %[[CONST_0]] to %[[CONST_1]] step %[[CONST_2]] iter_args(%arg1 = %[[VAR_4]]) -> (si32) : i32 {
// CHECK:       %[[VAR_8:.*]] = hwarith.cast %arg0 : (i32) -> si32
// CHECK:       %[[VAR_9:.*]] = coredsl.cast %[[VAR_8]] : si32 to ui5
// CHECK:       %[[VAR_10:.*]] = coredsl.get @MERGEDX[%[[VAR_1]] : ui5] : ui32
// CHECK:       %[[VAR_11:.*]] = coredsl.bitextract %[[VAR_10]][%[[VAR_9]] : ui5, 7:0] : (ui32) -> ui8
// CHECK:       %[[VAR_12:.*]] = coredsl.cast %[[VAR_11]] : ui8 to si8
// CHECK:       %[[VAR_13:.*]] = coredsl.cast %[[VAR_8]] : si32 to ui5
// CHECK:       %[[VAR_14:.*]] = coredsl.get @MERGEDX[%[[VAR_0]] : ui5] : ui32
// CHECK:       %[[VAR_15:.*]] = coredsl.bitextract %[[VAR_14]][%[[VAR_13]] : ui5, 7:0] : (ui32) -> ui8
// CHECK:       %[[VAR_16:.*]] = coredsl.cast %[[VAR_15]] : ui8 to si8
// CHECK:       %[[VAR_17:.*]] = hwarith.mul %[[VAR_12]], %[[VAR_16]] : (si8, si8) -> si16
// CHECK:       %[[VAR_18:.*]] = hwarith.add %arg1, %[[VAR_17]] : (si32, si16) -> si33
// CHECK:       %[[VAR_19:.*]] = coredsl.cast %[[VAR_18]] : si33 to si32
// CHECK:       scf.yield %[[VAR_19]] : si32
// CHECK:     }
// CHECK:     %[[VAR_6:.*]] = coredsl.cast %[[VAR_5]] : si32 to ui32
// CHECK:     coredsl.set @MERGEDX[%[[VAR_2]] : ui5] = %[[VAR_6]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @ijmp(%TREENAIL_WAS_HERE_offset_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1111011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_offset_11_0 : ui12 to ui12
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[VAR_3:.*]] = coredsl.get @MERGEDX[%[[VAR_1]] : ui5] : ui32
// CHECK:     %[[VAR_4:.*]] = hwarith.add %[[VAR_3]], %[[VAR_0]] : (ui32, ui12) -> ui33
// CHECK:     %[[VAR_5:.*]] = coredsl.cast %[[VAR_4]] : ui33 to ui32
// CHECK:     %[[VAR_6:.*]] = coredsl.get @MERGEDMEM[%[[VAR_5]] : ui32, 3:0] : ui32
// CHECK:     coredsl.set @MERGEDPC = %[[VAR_6]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @sbox("0000100", "00000", %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "000", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0101011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.get @MERGEDX[%[[VAR_0]] : ui5] : ui32
// CHECK:     %[[VAR_3:.*]] = coredsl.bitextract %[[VAR_2]][7:0] : (ui32) -> ui8
// CHECK:     %[[VAR_4:.*]] = coredsl.get @MERGED4SBOX[%[[VAR_3]] : ui8] : ui8
// CHECK:     %[[VAR_5:.*]] = coredsl.cast %[[VAR_4]] : ui8 to ui32
// CHECK:     coredsl.set @MERGEDX[%[[VAR_1]] : ui5] = %[[VAR_5]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @sparkle_ell("0000010", %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "111", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1111011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[VAR_3:.*]] = coredsl.get @MERGEDX[%[[VAR_1]] : ui5] : ui32
// CHECK:     %[[VAR_4:.*]] = coredsl.get @MERGEDX[%[[VAR_0]] : ui5] : ui32
// CHECK:     %[[VAR_5:.*]] = coredsl.xor %[[VAR_3]], %[[VAR_4]] : ui32, ui32
// CHECK:     %[[VAR_6:.*]] = func.call @MERGED5ELL(%[[VAR_5]]) : (ui32) -> ui32
// CHECK:     coredsl.set @MERGEDX[%[[VAR_2]] : ui5] = %[[VAR_6]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @sparkle_rcon("0000", %TREENAIL_WAS_HERE_imm_2_0 : ui3, %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "110", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1111011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_imm_2_0 : ui3 to ui3
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[VAR_3:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[VAR_4:.*]] = coredsl.get @MERGEDX[%[[VAR_2]] : ui5] : ui32
// CHECK:     %[[VAR_5:.*]] = coredsl.get @MERGED5RCON[%[[VAR_0]] : ui3] : ui32
// CHECK:     %[[VAR_6:.*]] = coredsl.xor %[[VAR_4]], %[[VAR_5]] : ui32, ui32
// CHECK:     coredsl.set @MERGEDX[%[[VAR_3]] : ui5] = %[[VAR_6]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @sparkle_whole_enci_x("1000", %TREENAIL_WAS_HERE_imm_2_0 : ui3, %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "110", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1111011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_imm_2_0 : ui3 to ui3
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[VAR_3:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[VAR_4:.*]] = coredsl.get @MERGEDX[%[[VAR_2]] : ui5] : ui32
// CHECK:     %[[VAR_5:.*]] = coredsl.get @MERGEDX[%[[VAR_1]] : ui5] : ui32
// CHECK:     %[[VAR_6:.*]] = coredsl.get @MERGED5RCON[%[[VAR_0]] : ui3] : ui32
// CHECK:     %[[CONST_3:.*]] = hw.constant 0 : i3
// CHECK:     %[[CONST_4:.*]] = hw.constant -4 : i3
// CHECK:     %[[CONST_5:.*]] = hw.constant 1 : i3
// CHECK:     %[[VAR_7]]:2 = scf.for %arg0 = %[[CONST_3]] to %[[CONST_4]] step %[[CONST_5]] iter_args(%arg1 = %[[VAR_4]], %arg2 = %[[VAR_5]]) -> (ui32, ui32) : i3 {
// CHECK:       %[[VAR_9:.*]] = hwarith.cast %arg0 : (i3) -> ui3
// CHECK:       %[[VAR_10:.*]] = coredsl.cast %[[VAR_9]] : ui3 to ui2
// CHECK:       %[[VAR_11:.*]] = coredsl.get @MERGED5ROT_0[%[[VAR_10]] : ui2] : ui8
// CHECK:       %[[VAR_12:.*]] = func.call @MERGED5ROTR(%arg2, %[[VAR_11]]) : (ui32, ui8) -> ui32
// CHECK:       %[[VAR_13:.*]] = hwarith.add %arg1, %[[VAR_12]] : (ui32, ui32) -> ui33
// CHECK:       %[[VAR_14:.*]] = coredsl.cast %[[VAR_13]] : ui33 to ui32
// CHECK:       %[[VAR_15:.*]] = coredsl.cast %[[VAR_9]] : ui3 to ui2
// CHECK:       %[[VAR_16:.*]] = coredsl.get @MERGED5ROT_1[%[[VAR_15]] : ui2] : ui8
// CHECK:       %[[VAR_17:.*]] = func.call @MERGED5ROTR(%[[VAR_14]], %[[VAR_16]]) : (ui32, ui8) -> ui32
// CHECK:       %[[VAR_18:.*]] = coredsl.xor %arg2, %[[VAR_17]] : ui32, ui32
// CHECK:       %[[VAR_19:.*]] = coredsl.xor %[[VAR_14]], %[[VAR_6]] : ui32, ui32
// CHECK:       scf.yield %[[VAR_19]], %[[VAR_18]] : ui32, ui32
// CHECK:     }
// CHECK:     coredsl.set @MERGEDX[%[[VAR_3]] : ui5] = %[[VAR_7]]#0 : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @sparkle_whole_enci_y("1001", %TREENAIL_WAS_HERE_imm_2_0 : ui3, %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "110", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1111011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_imm_2_0 : ui3 to ui3
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK:     %[[VAR_3:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[VAR_4:.*]] = coredsl.get @MERGEDX[%[[VAR_2]] : ui5] : ui32
// CHECK:     %[[VAR_5:.*]] = coredsl.get @MERGEDX[%[[VAR_1]] : ui5] : ui32
// CHECK:     %[[VAR_6:.*]] = coredsl.get @MERGED5RCON[%[[VAR_0]] : ui3] : ui32
// CHECK:     %[[CONST_6:.*]] = hw.constant 0 : i3
// CHECK:     %[[CONST_7:.*]] = hw.constant -4 : i3
// CHECK:     %[[CONST_8:.*]] = hw.constant 1 : i3
// CHECK:     %[[VAR_7]]:2 = scf.for %arg0 = %[[CONST_6]] to %[[CONST_7]] step %[[CONST_8]] iter_args(%arg1 = %[[VAR_4]], %arg2 = %[[VAR_5]]) -> (ui32, ui32) : i3 {
// CHECK:       %[[VAR_9:.*]] = hwarith.cast %arg0 : (i3) -> ui3
// CHECK:       %[[VAR_10:.*]] = coredsl.cast %[[VAR_9]] : ui3 to ui2
// CHECK:       %[[VAR_11:.*]] = coredsl.get @MERGED5ROT_0[%[[VAR_10]] : ui2] : ui8
// CHECK:       %[[VAR_12:.*]] = func.call @MERGED5ROTR(%arg2, %[[VAR_11]]) : (ui32, ui8) -> ui32
// CHECK:       %[[VAR_13:.*]] = hwarith.add %arg1, %[[VAR_12]] : (ui32, ui32) -> ui33
// CHECK:       %[[VAR_14:.*]] = coredsl.cast %[[VAR_13]] : ui33 to ui32
// CHECK:       %[[VAR_15:.*]] = coredsl.cast %[[VAR_9]] : ui3 to ui2
// CHECK:       %[[VAR_16:.*]] = coredsl.get @MERGED5ROT_1[%[[VAR_15]] : ui2] : ui8
// CHECK:       %[[VAR_17:.*]] = func.call @MERGED5ROTR(%[[VAR_14]], %[[VAR_16]]) : (ui32, ui8) -> ui32
// CHECK:       %[[VAR_18:.*]] = coredsl.xor %arg2, %[[VAR_17]] : ui32, ui32
// CHECK:       %[[VAR_19:.*]] = coredsl.xor %[[VAR_14]], %[[VAR_6]] : ui32, ui32
// CHECK:       scf.yield %[[VAR_19]], %[[VAR_18]] : ui32, ui32
// CHECK:     }
// CHECK:     coredsl.set @MERGEDX[%[[VAR_3]] : ui5] = %[[VAR_7]]#1 : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.instruction @sqrt_decoupled("0000110", "00000", %TREENAIL_WAS_HERE_rs_4_0 : ui5, "000", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0001011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rs_4_0 : ui5 to ui5
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.get @MERGEDX[%[[VAR_0]] : ui5] : ui32
// CHECK:     %[[VAR_3:.*]] = hwarith.constant 1073741824 : ui31
// CHECK:     %[[VAR_4:.*]] = coredsl.cast %[[VAR_3]] : ui31 to ui32
// CHECK:     %[[VAR_5:.*]] = hwarith.constant 0 : ui1
// CHECK:     %[[VAR_6:.*]] = coredsl.cast %[[VAR_5]] : ui1 to ui32
// CHECK:     %[[CONST_9:.*]] = hw.constant 0 : i32
// CHECK:     %[[CONST_10:.*]] = hw.constant 32 : i32
// CHECK:     %[[CONST_11:.*]] = hw.constant 1 : i32
// CHECK:     %[[VAR_7:.*]]:3 = scf.for %arg0 = %[[CONST_9]] to %[[CONST_10]] step %[[CONST_11]] iter_args(%arg1 = %[[VAR_2]], %arg2 = %[[VAR_6]], %arg3 = %[[VAR_4]]) -> (ui32, ui32, ui32) : i32 {
// CHECK:       %[[VAR_12:.*]] = hwarith.cast %arg0 : (i32) -> si32
// CHECK:       %[[VAR_13:.*]] = hwarith.add %arg2, %arg3 : (ui32, ui32) -> ui33
// CHECK:       %[[VAR_14:.*]] = coredsl.cast %[[VAR_13]] : ui33 to ui32
// CHECK:       %[[VAR_15:.*]] = hwarith.icmp ge %arg1, %[[VAR_14]] : ui32, ui32
// CHECK:       %[[VAR_17:.*]]:2 = scf.if %[[VAR_15]] -> (ui32, ui32) {
// CHECK:         %[[VAR_22:.*]] = hwarith.sub %arg1, %[[VAR_14]] : (ui32, ui32) -> si33
// CHECK:         %[[VAR_23:.*]] = coredsl.cast %[[VAR_22]] : si33 to ui32
// CHECK:         %[[VAR_24:.*]] = hwarith.add %[[VAR_14]], %arg3 : (ui32, ui32) -> ui33
// CHECK:         %[[VAR_25:.*]] = coredsl.cast %[[VAR_24]] : ui33 to ui32
// CHECK:         scf.yield %[[VAR_23]], %[[VAR_25]] : ui32, ui32
// CHECK:       } else {
// CHECK:         scf.yield %arg1, %arg2 : ui32, ui32
// CHECK:       }
// CHECK:       %[[VAR_18:.*]] = hwarith.constant 1 : ui1
// CHECK:       %[[VAR_19:.*]] = coredsl.shift_left %[[VAR_17]]#0, %[[VAR_18]] : ui32, ui1
// CHECK:       %[[VAR_20:.*]] = hwarith.constant 1 : ui1
// CHECK:       %[[VAR_21:.*]] = coredsl.shift_right %arg3, %[[VAR_20]] : ui32, ui1
// CHECK:       scf.yield %[[VAR_19]], %[[VAR_17]]#1, %[[VAR_21]] : ui32, ui32, ui32
// CHECK:     }
// CHECK:     %[[VAR_8:.*]] = hwarith.icmp gt %[[VAR_7]]#0, %[[VAR_7]]#1 : ui32, ui32
// CHECK:     %[[VAR_10:.*]] = scf.if %[[VAR_8]] -> (ui32) {
// CHECK:       %[[VAR_11:.*]] = hwarith.constant 1 : ui1
// CHECK:       %[[VAR_12:.*]] = hwarith.add %[[VAR_7]]#1, %[[VAR_11]] : (ui32, ui1) -> ui33
// CHECK:       %[[VAR_13:.*]] = coredsl.cast %[[VAR_12]] : ui33 to ui32
// CHECK:       scf.yield %[[VAR_13]] : ui32
// CHECK:     } else {
// CHECK:       scf.yield %[[VAR_7]]#1 : ui32
// CHECK:     }
// CHECK:     coredsl.spawn {
// CHECK:       coredsl.set @MERGEDX[%[[VAR_1]] : ui5] = %[[VAR_10]] : ui32
// CHECK:       coredsl.end
// CHECK:     }
// CHECK:   }
// CHECK:   coredsl.instruction @setup_zol(%TREENAIL_WAS_HERE_uimmL_11_0 : ui12, %TREENAIL_WAS_HERE_uimmS_4_0 : ui5, "101", "00000", "0001011"){
// CHECK:     %[[VAR_0:.*]] = coredsl.cast %TREENAIL_WAS_HERE_uimmL_11_0 : ui12 to ui12
// CHECK:     %[[VAR_1:.*]] = coredsl.cast %TREENAIL_WAS_HERE_uimmS_4_0 : ui5 to ui5
// CHECK:     %[[VAR_2:.*]] = coredsl.get @MERGEDPC : ui32
// CHECK:     %[[VAR_3:.*]] = hwarith.constant 4 : ui3
// CHECK:     %[[VAR_4:.*]] = hwarith.add %[[VAR_2]], %[[VAR_3]] : (ui32, ui3) -> ui33
// CHECK:     %[[VAR_5:.*]] = coredsl.cast %[[VAR_4]] : ui33 to ui32
// CHECK:     coredsl.set @MERGED7ZOLSTARTPC = %[[VAR_5]] : ui32
// CHECK:     %[[VAR_6:.*]] = coredsl.get @MERGEDPC : ui32
// CHECK:     %[[VAR_7:.*]] = hwarith.constant 0 : ui1
// CHECK:     %[[VAR_8:.*]] = coredsl.concat %[[VAR_1]], %[[VAR_7]] : ui5, ui1
// CHECK:     %[[VAR_9:.*]] = hwarith.add %[[VAR_6]], %[[VAR_8]] : (ui32, ui6) -> ui33
// CHECK:     %[[VAR_10:.*]] = coredsl.cast %[[VAR_9]] : ui33 to ui32
// CHECK:     coredsl.set @MERGED7ZOLENDPC = %[[VAR_10]] : ui32
// CHECK:     %[[VAR_11:.*]] = coredsl.cast %[[VAR_0]] : ui12 to ui32
// CHECK:     coredsl.set @MERGED7ZOLCOUNTER = %[[VAR_11]] : ui32
// CHECK:     coredsl.end
// CHECK:   }
// CHECK:   coredsl.always @zol {
// CHECK:     %[[VAR_0:.*]] = coredsl.get @MERGED7ZOLCOUNTER : ui32
// CHECK:     %[[VAR_1:.*]] = hwarith.constant 0 : ui1
// CHECK:     %[[VAR_2:.*]] = hwarith.icmp ne %[[VAR_0]], %[[VAR_1]] : ui32, ui1
// CHECK:     %[[VAR_2_1:.*]] = hwarith.cast %[[VAR_2]] : (i1) -> ui1
// CHECK:     %[[VAR_3:.*]] = coredsl.get @MERGED7ZOLENDPC : ui32
// CHECK:     %[[VAR_4:.*]] = coredsl.get @MERGEDPC : ui32
// CHECK:     %[[VAR_5:.*]] = hwarith.icmp eq %[[VAR_3]], %[[VAR_4]] : ui32, ui32
// CHECK:     %[[VAR_5_1:.*]] = hwarith.cast %[[VAR_5]] : (i1) -> ui1
// CHECK:     %[[VAR_6:.*]] = coredsl.and %[[VAR_2_1]], %[[VAR_5_1]] : ui1, ui1
// CHECK:     %[[VAR_7:.*]] = coredsl.cast %[[VAR_6]] : ui1 to i1
// CHECK:     scf.if %[[VAR_7]] {
// CHECK:       %[[VAR_8:.*]] = coredsl.get @MERGED7ZOLSTARTPC : ui32
// CHECK:       coredsl.set @MERGEDPC = %[[VAR_8]] : ui32
// CHECK:       %[[VAR_9:.*]] = hwarith.constant 1 : ui1
// CHECK:       %[[VAR_10:.*]] = hwarith.sub %[[VAR_0]], %[[VAR_9]] : (ui32, ui1) -> si33
// CHECK:       %[[VAR_11:.*]] = coredsl.cast %[[VAR_10]] : si33 to ui32
// CHECK:       coredsl.set @MERGED7ZOLCOUNTER = %[[VAR_11]] : ui32
// CHECK:     }
// CHECK:     coredsl.end
// CHECK:   }
// CHECK: }

coredsl.isax "autoinc" {
  coredsl.register core_x @X[32] : ui32
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8
  coredsl.register local @ADDR : ui32
  coredsl.instruction @set_addr("000000000000", %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "000", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0001011") {
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    coredsl.set @ADDR = %0 : ui32
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
    %5 = hwarith.icmp eq %3, %4 : ui32, si5
    scf.if %5 {
      %7 = coredsl.get @PC : ui32
      %8 = hwarith.add %7, %2 : (ui32, si13) -> si34
      %9 = coredsl.cast %8 : si34 to ui32
      coredsl.set @PC = %9 : ui32
    }
    coredsl.end
  }
}
coredsl.isax "dotprod" {
  coredsl.register core_x @X[32] : ui32
  coredsl.instruction @DOTP("0001001", %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "000", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0001011") {
    %rs2 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = hwarith.constant 0 : ui1
    %1 = coredsl.cast %0 : ui1 to si32
    %2 = hw.constant 0 : i32
    %3 = hw.constant 32 : i32
    %4 = hw.constant 8 : i32
    %5 = scf.for %5 = %2 to %3 step %4 iter_args(%8 = %1) -> (si32) : i32 {
      %7 = hwarith.cast %5 : (i32) -> si32
      %10 = coredsl.cast %7 : si32 to ui5
      %11 = coredsl.get @X[%rs1 : ui5] : ui32
      %9 = coredsl.bitextract %11[%10 : ui5, 7:0] : (ui32) -> ui8
      %12 = coredsl.cast %9 : ui8 to si8
      %14 = coredsl.cast %7 : si32 to ui5
      %15 = coredsl.get @X[%rs2 : ui5] : ui32
      %13 = coredsl.bitextract %15[%14 : ui5, 7:0] : (ui32) -> ui8
      %16 = coredsl.cast %13 : ui8 to si8
      %17 = hwarith.mul %12, %16 : (si8, si8) -> si16
      %18 = hwarith.add %8, %17 : (si32, si16) -> si33
      %19 = coredsl.cast %18 : si33 to si32
      scf.yield %19 : si32
    }
    %6 = coredsl.cast %5 : si32 to ui32
    coredsl.set @X[%rd : ui5] = %6 : ui32
    coredsl.end
  }
}
coredsl.isax "indirectjmp" {
  coredsl.register core_x @X[32] : ui32
  coredsl.register core_pc @PC : ui32
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8
  coredsl.instruction @ijmp(%TREENAIL_WAS_HERE_offset_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1111011") {
    %offset = coredsl.cast %TREENAIL_WAS_HERE_offset_11_0 : ui12 to ui12
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = hwarith.add %0, %offset : (ui32, ui12) -> ui33
    %2 = coredsl.cast %1 : ui33 to ui32
    %3 = coredsl.get @MEM[%2 : ui32, 3:0] : ui32
    coredsl.set @PC = %3 : ui32
    coredsl.end
  }
}
coredsl.isax "sbox" {
  coredsl.register core_x @X[32] : ui32
  coredsl.register local const @SBOX[256] = [99, 124, 119, 123, 242, 107, 111, 197, 48, 1, 103, 43, 254, 215, 171, 118, 202, 130, 201, 125, 250, 89, 71, 240, 173, 212, 162, 175, 156, 164, 114, 192, 183, 253, 147, 38, 54, 63, 247, 204, 52, 165, 229, 241, 113, 216, 49, 21, 4, 199, 35, 195, 24, 150, 5, 154, 7, 18, 128, 226, 235, 39, 178, 117, 9, 131, 44, 26, 27, 110, 90, 160, 82, 59, 214, 179, 41, 227, 47, 132, 83, 209, 0, 237, 32, 252, 177, 91, 106, 203, 190, 57, 74, 76, 88, 207, 208, 239, 170, 251, 67, 77, 51, 133, 69, 249, 2, 127, 80, 60, 159, 168, 81, 163, 64, 143, 146, 157, 56, 245, 188, 182, 218, 33, 16, 255, 243, 210, 205, 12, 19, 236, 95, 151, 68, 23, 196, 167, 126, 61, 100, 93, 25, 115, 96, 129, 79, 220, 34, 42, 144, 136, 70, 238, 184, 20, 222, 94, 11, 219, 224, 50, 58, 10, 73, 6, 36, 92, 194, 211, 172, 98, 145, 149, 228, 121, 231, 200, 55, 109, 141, 213, 78, 169, 108, 86, 244, 234, 101, 122, 174, 8, 186, 120, 37, 46, 28, 166, 180, 198, 232, 221, 116, 31, 75, 189, 139, 138, 112, 62, 181, 102, 72, 3, 246, 14, 97, 53, 87, 185, 134, 193, 29, 158, 225, 248, 152, 17, 105, 217, 142, 148, 155, 30, 135, 233, 206, 85, 40, 223, 140, 161, 137, 13, 191, 230, 66, 104, 65, 153, 45, 15, 176, 84, 187, 22] : ui8
  coredsl.instruction @sbox("0000100", "00000", %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "000", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0101011") {
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %2 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.bitextract %2[7:0] : (ui32) -> ui8
    %0 = coredsl.get @SBOX[%1 : ui8] : ui8
    %3 = coredsl.cast %0 : ui8 to ui32
    coredsl.set @X[%rd : ui5] = %3 : ui32
    coredsl.end
  }
}
coredsl.isax "sparkle" {
  coredsl.register core_x @X[32] : ui32
  coredsl.register local const @ROT_0[4] = [31, 17, 0, 24] : ui8
  coredsl.register local const @ROT_1[4] = [24, 17, 31, 16] : ui8
  coredsl.register local const @RCON[8] = [3084996962, 3211876480, 951376470, 844003128, 3138487787, 1333558103, 3485442504, 3266521405] : ui32
  func.func @ROTR(%x : ui32, %shamt : ui8) -> ui32 {
    %0 = coredsl.shift_right %x, %shamt : ui32, ui8
    %1 = hwarith.constant 32 : ui6
    %2 = hwarith.sub %1, %shamt : (ui6, ui8) -> si9
    %3 = coredsl.shift_left %x, %2 : ui32, si9
    %4 = coredsl.or %0, %3 : ui32, ui32
    return %4 : ui32
  }
  func.func @ELL(%x : ui32) -> ui32 {
    %0 = hwarith.constant 16 : ui5
    %1 = coredsl.shift_left %x, %0 : ui32, ui5
    %2 = coredsl.xor %x, %1 : ui32, ui32
    %3 = hwarith.constant 16 : ui5
    %4 = coredsl.cast %3 : ui5 to ui8
    %5 = func.call @ROTR(%2, %4) : (ui32, ui8) -> ui32
    return %5 : ui32
  }
  coredsl.instruction @sparkle_ell("0000010", %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "111", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1111011") {
    %rs2 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    %2 = coredsl.xor %0, %1 : ui32, ui32
    %3 = func.call @ELL(%2) : (ui32) -> ui32
    coredsl.set @X[%rd : ui5] = %3 : ui32
    coredsl.end
  }
  coredsl.instruction @sparkle_rcon("0000", %TREENAIL_WAS_HERE_imm_2_0 : ui3, %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "110", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1111011") {
    %imm = coredsl.cast %TREENAIL_WAS_HERE_imm_2_0 : ui3 to ui3
    %rs2 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @RCON[%imm : ui3] : ui32
    %2 = coredsl.xor %0, %1 : ui32, ui32
    coredsl.set @X[%rd : ui5] = %2 : ui32
    coredsl.end
  }
  coredsl.instruction @sparkle_whole_enci_x("1000", %TREENAIL_WAS_HERE_imm_2_0 : ui3, %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "110", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1111011") {
    %imm = coredsl.cast %TREENAIL_WAS_HERE_imm_2_0 : ui3 to ui3
    %rs2 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    %2 = coredsl.get @RCON[%imm : ui3] : ui32
    %3 = hw.constant 0 : i3
    %4 = hw.constant 4 : i3
    %5 = hw.constant 1 : i3
    %6, %7 = scf.for %6 = %3 to %4 step %5 iter_args(%9 = %0, %10 = %1) -> (ui32, ui32) : i3 {
      %8 = hwarith.cast %6 : (i3) -> ui3
      %12 = coredsl.cast %8 : ui3 to ui2
      %11 = coredsl.get @ROT_0[%12 : ui2] : ui8
      %13 = func.call @ROTR(%10, %11) : (ui32, ui8) -> ui32
      %14 = hwarith.add %9, %13 : (ui32, ui32) -> ui33
      %15 = coredsl.cast %14 : ui33 to ui32
      %17 = coredsl.cast %8 : ui3 to ui2
      %16 = coredsl.get @ROT_1[%17 : ui2] : ui8
      %18 = func.call @ROTR(%15, %16) : (ui32, ui8) -> ui32
      %19 = coredsl.xor %10, %18 : ui32, ui32
      %20 = coredsl.xor %15, %2 : ui32, ui32
      scf.yield %20, %19 : ui32, ui32
    }
    coredsl.set @X[%rd : ui5] = %6 : ui32
    coredsl.end
  }
  coredsl.instruction @sparkle_whole_enci_y("1001", %TREENAIL_WAS_HERE_imm_2_0 : ui3, %TREENAIL_WAS_HERE_rs2_4_0 : ui5, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "110", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "1111011") {
    %imm = coredsl.cast %TREENAIL_WAS_HERE_imm_2_0 : ui3 to ui3
    %rs2 = coredsl.cast %TREENAIL_WAS_HERE_rs2_4_0 : ui5 to ui5
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    %2 = coredsl.get @RCON[%imm : ui3] : ui32
    %3 = hw.constant 0 : i3
    %4 = hw.constant 4 : i3
    %5 = hw.constant 1 : i3
    %6, %7 = scf.for %6 = %3 to %4 step %5 iter_args(%9 = %0, %10 = %1) -> (ui32, ui32) : i3 {
      %8 = hwarith.cast %6 : (i3) -> ui3
      %12 = coredsl.cast %8 : ui3 to ui2
      %11 = coredsl.get @ROT_0[%12 : ui2] : ui8
      %13 = func.call @ROTR(%10, %11) : (ui32, ui8) -> ui32
      %14 = hwarith.add %9, %13 : (ui32, ui32) -> ui33
      %15 = coredsl.cast %14 : ui33 to ui32
      %17 = coredsl.cast %8 : ui3 to ui2
      %16 = coredsl.get @ROT_1[%17 : ui2] : ui8
      %18 = func.call @ROTR(%15, %16) : (ui32, ui8) -> ui32
      %19 = coredsl.xor %10, %18 : ui32, ui32
      %20 = coredsl.xor %15, %2 : ui32, ui32
      scf.yield %20, %19 : ui32, ui32
    }
    coredsl.set @X[%rd : ui5] = %7 : ui32
    coredsl.end
  }
}
coredsl.isax "sqrt" {
  coredsl.register core_x @X[32] : ui32
  coredsl.instruction @sqrt_decoupled("0000110", "00000", %TREENAIL_WAS_HERE_rs_4_0 : ui5, "000", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0001011") {
    %rs = coredsl.cast %TREENAIL_WAS_HERE_rs_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @X[%rs : ui5] : ui32
    %1 = hwarith.constant 1073741824 : ui31
    %2 = coredsl.cast %1 : ui31 to ui32
    %3 = hwarith.constant 0 : ui1
    %4 = coredsl.cast %3 : ui1 to ui32
    %5 = hw.constant 0 : i32
    %6 = hw.constant 32 : i32
    %7 = hw.constant 1 : i32
    %8, %9, %10 = scf.for %8 = %5 to %6 step %7 iter_args(%11 = %0, %12 = %4, %13 = %2) -> (ui32, ui32, ui32) : i32 {
      %10 = hwarith.cast %8 : (i32) -> si32
      %14 = hwarith.add %12, %13 : (ui32, ui32) -> ui33
      %15 = coredsl.cast %14 : ui33 to ui32
      %17 = hwarith.icmp ge %11, %15 : ui32, ui32
      %18, %19 = scf.if %17 -> (ui32, ui32) {
        %18 = hwarith.sub %11, %15 : (ui32, ui32) -> si33
        %19 = coredsl.cast %18 : si33 to ui32
        %20 = hwarith.add %15, %13 : (ui32, ui32) -> ui33
        %21 = coredsl.cast %20 : ui33 to ui32
        scf.yield %19, %21 : ui32, ui32
      } else {
        scf.yield %11, %12 : ui32, ui32
      }
      %20 = hwarith.constant 1 : ui1
      %21 = coredsl.shift_left %18, %20 : ui32, ui1
      %22 = hwarith.constant 1 : ui1
      %23 = coredsl.shift_right %13, %22 : ui32, ui1
      scf.yield %21, %19, %23 : ui32, ui32, ui32
    }
    %12 = hwarith.icmp gt %8, %9 : ui32, ui32
    %13 = scf.if %12 -> (ui32) {
      %13 = hwarith.constant 1 : ui1
      %14 = hwarith.add %9, %13 : (ui32, ui1) -> ui33
      %15 = coredsl.cast %14 : ui33 to ui32
      scf.yield %15 : ui32
    } else {
      scf.yield %9 : ui32
    }
    coredsl.spawn {
      coredsl.set @X[%rd : ui5] = %13 : ui32
      coredsl.end
    }
  }
}
coredsl.isax "zol" {
  coredsl.register core_x @X[32] : ui32
  coredsl.register core_pc @PC : ui32
  coredsl.register local @ZOLSTARTPC : ui32
  coredsl.register local @ZOLENDPC : ui32
  coredsl.register local @ZOLCOUNTER : ui32
  coredsl.instruction @setup_zol(%TREENAIL_WAS_HERE_uimmL_11_0 : ui12, %TREENAIL_WAS_HERE_uimmS_4_0 : ui5, "101", "00000", "0001011") {
    %uimmL = coredsl.cast %TREENAIL_WAS_HERE_uimmL_11_0 : ui12 to ui12
    %uimmS = coredsl.cast %TREENAIL_WAS_HERE_uimmS_4_0 : ui5 to ui5
    %0 = coredsl.get @PC : ui32
    %1 = hwarith.constant 4 : ui3
    %2 = hwarith.add %0, %1 : (ui32, ui3) -> ui33
    %3 = coredsl.cast %2 : ui33 to ui32
    coredsl.set @ZOLSTARTPC = %3 : ui32
    %4 = coredsl.get @PC : ui32
    %5 = hwarith.constant 0 : ui1
    %6 = coredsl.concat %uimmS, %5 : ui5, ui1
    %7 = hwarith.add %4, %6 : (ui32, ui6) -> ui33
    %8 = coredsl.cast %7 : ui33 to ui32
    coredsl.set @ZOLENDPC = %8 : ui32
    %9 = coredsl.cast %uimmL : ui12 to ui32
    coredsl.set @ZOLCOUNTER = %9 : ui32
    coredsl.end
  }
  coredsl.always @zol {
    %0 = coredsl.get @ZOLCOUNTER : ui32
    %1 = hwarith.constant 0 : ui1
    %2 = hwarith.icmp ne %0, %1 : ui32, ui1
    %21 = hwarith.cast %2 : (i1) -> ui1
    %3 = coredsl.get @ZOLENDPC : ui32
    %4 = coredsl.get @PC : ui32
    %5 = hwarith.icmp eq %3, %4 : ui32, ui32
    %51 = hwarith.cast %5 : (i1) -> ui1
    %6 = coredsl.and %21, %51 : ui1, ui1
    %7 = coredsl.cast %6 : ui1 to i1
    scf.if %7 {
      %8 = coredsl.get @ZOLSTARTPC : ui32
      coredsl.set @PC = %8 : ui32
      %9 = hwarith.constant 1 : ui1
      %10 = hwarith.sub %0, %9 : (ui32, ui1) -> si33
      %11 = coredsl.cast %10 : si33 to ui32
      coredsl.set @ZOLCOUNTER = %11 : ui32
    }
    coredsl.end
  }
}
