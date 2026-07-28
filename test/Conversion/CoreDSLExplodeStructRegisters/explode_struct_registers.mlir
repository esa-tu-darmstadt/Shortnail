// RUN: shortnail-opt %s -coredsl-explode-struct-registers -canonicalize | shortnail-opt | FileCheck %s

coredsl.isax "StructRegisters" {
  coredsl.register local @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
  coredsl.register local @NESTED_STRUCT_REG : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
  coredsl.register local @TRIPLE_NESTED_REG : !hw.struct<internalStruct: !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>, intVal: ui32>
  coredsl.register local @SCALAR_REG1 : ui32
  coredsl.register local @SCALAR_REG2 : ui32
  coredsl.register local @STRUCT_REGS[32] : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>

  coredsl.instruction @StructRegDirectStore {lil.enc_immediates = [[["%TREENAIL_WAS_HERE_imm_11_0", 11, 0, 0, "imm"]], [["%TREENAIL_WAS_HERE_rs1_4_0", 4, 0, 0, "rs1"]], [["%TREENAIL_WAS_HERE_rd_4_0", 4, 0, 0, "rd"]]]} (%TREENAIL_WAS_HERE_imm_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0000011") {
    %imm = coredsl.cast %TREENAIL_WAS_HERE_imm_11_0 : ui12 to ui12
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
    %1 = coredsl.cast %rs1 : ui5 to ui32
    %2 = hw.struct_inject %0["x"], %1 : !hw.struct<x: ui32, y: ui32>
    coredsl.set @STRUCT_REG = %2 : !hw.struct<x: ui32, y: ui32>
    %4 = coredsl.get @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
    %5 = hw.struct_extract %4["x"] : !hw.struct<x: ui32, y: ui32>
    %10 = hwarith.constant 7 : ui3
    %11 = coredsl.get @NESTED_STRUCT_REG : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %12 = hw.struct_extract %11["vec"] : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %13 = coredsl.cast %10 : ui3 to ui32
    %14 = hw.struct_inject %12["x"], %13 : !hw.struct<x: ui32, y: ui32>
    %15 = hw.struct_inject %11["vec"], %14 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    coredsl.set @NESTED_STRUCT_REG = %15 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %16 = hwarith.constant 255 : ui8
    %17 = coredsl.get @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
    %18 = hw.struct_extract %17["x"] : !hw.struct<x: ui32, y: ui32>
    %19 = coredsl.bitset %18[7:0] = %16 : (ui32, ui8) -> ui32
    %20 = hw.struct_inject %17["x"], %19 : !hw.struct<x: ui32, y: ui32>
    coredsl.set @STRUCT_REG = %20 : !hw.struct<x: ui32, y: ui32>
    %21 = hwarith.constant 0 : ui1
    %22 = coredsl.get @NESTED_STRUCT_REG : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %23 = hw.struct_extract %22["vec"] : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %24 = hw.struct_extract %23["y"] : !hw.struct<x: ui32, y: ui32>
    %25 = coredsl.cast %21 : ui1 to ui4
    %26 = coredsl.bitset %24[3:0] = %25 : (ui32, ui4) -> ui32
    %27 = hw.struct_inject %23["y"], %26 : !hw.struct<x: ui32, y: ui32>
    %28 = hw.struct_inject %22["vec"], %27 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    coredsl.set @NESTED_STRUCT_REG = %28 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %29 = coredsl.get @TRIPLE_NESTED_REG : !hw.struct<internalStruct: !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>, intVal: ui32>
    %30 = hw.struct_extract %29["internalStruct"] : !hw.struct<internalStruct: !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>, intVal: ui32>
    %31 = hwarith.constant -1 : si32
    %32 = hw.struct_inject %30["notNested"], %31 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %33 = hw.struct_extract %32["vec"] : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %34 = hw.struct_inject %33["x"], %26 : !hw.struct<x: ui32, y: ui32>
    %35 = hw.struct_inject %32["vec"], %34 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %36 = hw.struct_inject %29["internalStruct"], %35 : !hw.struct<internalStruct: !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>, intVal: ui32>
    coredsl.set @TRIPLE_NESTED_REG = %36 : !hw.struct<internalStruct: !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>, intVal: ui32>
    coredsl.end
  }
  coredsl.instruction @TransferStructToScalarReg{lil.enc_immediates = [[["%TREENAIL_WAS_HERE_imm_11_0", 11, 0, 0, "imm"]], [["%TREENAIL_WAS_HERE_rs1_4_0", 4, 0, 0, "rs1"]], [["%TREENAIL_WAS_HERE_rd_4_0", 4, 0, 0, "rd"]]]} (%TREENAIL_WAS_HERE_imm_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0000011") {
    %0 = coredsl.get @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
    %1 = hw.struct_extract %0["x"] : !hw.struct<x: ui32, y: ui32>
    %2 = hw.struct_extract %0["y"] : !hw.struct<x: ui32, y: ui32>
    %3 = hwarith.constant 1 : ui1
    %4 = hwarith.add %1, %3 : (ui32, ui1) -> ui33
    %5 = coredsl.cast %4 : ui33 to ui32
    coredsl.set @SCALAR_REG1 = %5 : ui32
    coredsl.set @SCALAR_REG2 = %2 : ui32
    coredsl.end
  }

  coredsl.instruction @StructArrays{lil.enc_immediates = [[["%TREENAIL_WAS_HERE_imm_11_0", 11, 0, 0, "imm"]], [["%TREENAIL_WAS_HERE_rs1_4_0", 4, 0, 0, "rs1"]], [["%TREENAIL_WAS_HERE_rd_4_0", 4, 0, 0, "rd"]]]} (%TREENAIL_WAS_HERE_imm_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0000011") {
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %21 = coredsl.get @NESTED_STRUCT_REG : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    coredsl.set @STRUCT_REGS[2] = %21 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %22 = hwarith.constant 10 : ui4
    %23 = coredsl.get @STRUCT_REGS[%rs1 : ui5] : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %24 = coredsl.cast %22 : ui4 to si32
    %25 = hw.struct_inject %23["notNested"], %24 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    coredsl.set @STRUCT_REGS[%rs1 : ui5] = %25 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    coredsl.end
  }
}

// CHECK-LABEL:   coredsl.isax "StructRegisters" {
// CHECK:           coredsl.register local @STRUCT_REG_x  : ui32
// CHECK:           coredsl.register local @STRUCT_REG_y  : ui32
// CHECK:           coredsl.register local @NESTED_STRUCT_REG_notNested  : si32
// CHECK:           coredsl.register local @NESTED_STRUCT_REG_vec_x  : ui32
// CHECK:           coredsl.register local @NESTED_STRUCT_REG_vec_y  : ui32
// CHECK:           coredsl.register local @TRIPLE_NESTED_REG_internalStruct_notNested  : si32
// CHECK:           coredsl.register local @TRIPLE_NESTED_REG_internalStruct_vec_x  : ui32
// CHECK:           coredsl.register local @TRIPLE_NESTED_REG_internalStruct_vec_y  : ui32
// CHECK:           coredsl.register local @TRIPLE_NESTED_REG_intVal  : ui32
// CHECK:           coredsl.register local @SCALAR_REG1  : ui32
// CHECK:           coredsl.register local @SCALAR_REG2  : ui32
// CHECK:           coredsl.register local @STRUCT_REGS_notNested[32]  : si32
// CHECK:           coredsl.register local @STRUCT_REGS_vec_x[32]  : ui32
// CHECK:           coredsl.register local @STRUCT_REGS_vec_y[32]  : ui32
// CHECK:           coredsl.instruction @StructRegDirectStore {lil.enc_immediates = {{\[\[}}["%[[VAL_0:.*]]", 11, 0, 0, "imm"]], {{\[\[}}"%[[VAL_1:.*]]", 4, 0, 0, "rs1"]], {{\[\[}}"%[[VAL_2:.*]]", 4, 0, 0, "rd"]]]}(%[[VAL_0]] : ui12, %[[VAL_1]] : ui5, "010", %[[VAL_2]] : ui5, "0000011"){
// CHECK:             %[[CONSTANT_0:.*]] = hwarith.constant -1 : si32
// CHECK:             %[[CONSTANT_1:.*]] = hwarith.constant 0 : ui1
// CHECK:             %[[CONSTANT_2:.*]] = hwarith.constant 255 : ui8
// CHECK:             %[[CONSTANT_3:.*]] = hwarith.constant 7 : ui3
// CHECK:             %[[CAST_0:.*]] = coredsl.cast %[[VAL_1]] : ui5 to ui5
// CHECK:             %[[GET_0:.*]] = coredsl.get @STRUCT_REG_x : ui32
// CHECK:             %[[GET_1:.*]] = coredsl.get @STRUCT_REG_y : ui32
// CHECK:             %[[CAST_1:.*]] = coredsl.cast %[[CAST_0]] : ui5 to ui32
// CHECK:             coredsl.set @STRUCT_REG_x = %[[CAST_1]] : ui32
// CHECK:             coredsl.set @STRUCT_REG_y = %[[GET_1]] : ui32
// CHECK:             %[[GET_2:.*]] = coredsl.get @STRUCT_REG_x : ui32
// CHECK:             %[[GET_3:.*]] = coredsl.get @STRUCT_REG_y : ui32
// CHECK:             %[[GET_4:.*]] = coredsl.get @NESTED_STRUCT_REG_notNested : si32
// CHECK:             %[[GET_5:.*]] = coredsl.get @NESTED_STRUCT_REG_vec_x : ui32
// CHECK:             %[[GET_6:.*]] = coredsl.get @NESTED_STRUCT_REG_vec_y : ui32
// CHECK:             %[[CAST_2:.*]] = coredsl.cast %[[CONSTANT_3]] : ui3 to ui32
// CHECK:             coredsl.set @NESTED_STRUCT_REG_notNested = %[[GET_4]] : si32
// CHECK:             coredsl.set @NESTED_STRUCT_REG_vec_x = %[[CAST_2]] : ui32
// CHECK:             coredsl.set @NESTED_STRUCT_REG_vec_y = %[[GET_6]] : ui32
// CHECK:             %[[GET_7:.*]] = coredsl.get @STRUCT_REG_x : ui32
// CHECK:             %[[GET_8:.*]] = coredsl.get @STRUCT_REG_y : ui32
// CHECK:             %[[BITSET_0:.*]] = coredsl.bitset %[[GET_7]][7:0] = %[[CONSTANT_2]] : (ui32, ui8) -> ui32
// CHECK:             coredsl.set @STRUCT_REG_x = %[[BITSET_0]] : ui32
// CHECK:             coredsl.set @STRUCT_REG_y = %[[GET_8]] : ui32
// CHECK:             %[[GET_9:.*]] = coredsl.get @NESTED_STRUCT_REG_notNested : si32
// CHECK:             %[[GET_10:.*]] = coredsl.get @NESTED_STRUCT_REG_vec_x : ui32
// CHECK:             %[[GET_11:.*]] = coredsl.get @NESTED_STRUCT_REG_vec_y : ui32
// CHECK:             %[[CAST_3:.*]] = coredsl.cast %[[CONSTANT_1]] : ui1 to ui4
// CHECK:             %[[BITSET_1:.*]] = coredsl.bitset %[[GET_11]][3:0] = %[[CAST_3]] : (ui32, ui4) -> ui32
// CHECK:             coredsl.set @NESTED_STRUCT_REG_notNested = %[[GET_9]] : si32
// CHECK:             coredsl.set @NESTED_STRUCT_REG_vec_x = %[[GET_10]] : ui32
// CHECK:             coredsl.set @NESTED_STRUCT_REG_vec_y = %[[BITSET_1]] : ui32
// CHECK:             %[[GET_12:.*]] = coredsl.get @TRIPLE_NESTED_REG_internalStruct_notNested : si32
// CHECK:             %[[GET_13:.*]] = coredsl.get @TRIPLE_NESTED_REG_internalStruct_vec_x : ui32
// CHECK:             %[[GET_14:.*]] = coredsl.get @TRIPLE_NESTED_REG_internalStruct_vec_y : ui32
// CHECK:             %[[GET_15:.*]] = coredsl.get @TRIPLE_NESTED_REG_intVal : ui32
// CHECK:             coredsl.set @TRIPLE_NESTED_REG_internalStruct_notNested = %[[CONSTANT_0]] : si32
// CHECK:             coredsl.set @TRIPLE_NESTED_REG_internalStruct_vec_x = %[[BITSET_1]] : ui32
// CHECK:             coredsl.set @TRIPLE_NESTED_REG_internalStruct_vec_y = %[[GET_14]] : ui32
// CHECK:             coredsl.set @TRIPLE_NESTED_REG_intVal = %[[GET_15]] : ui32
// CHECK:             coredsl.end
// CHECK:           }
// CHECK:           coredsl.instruction @TransferStructToScalarReg {lil.enc_immediates = {{\[\[}}["%[[VAL_3:.*]]", 11, 0, 0, "imm"]], {{\[\[}}"%[[VAL_4:.*]]", 4, 0, 0, "rs1"]], {{\[\[}}"%[[VAL_5:.*]]", 4, 0, 0, "rd"]]]}(%[[VAL_3]] : ui12, %[[VAL_4]] : ui5, "010", %[[VAL_5]] : ui5, "0000011"){
// CHECK:             %[[CONSTANT_4:.*]] = hwarith.constant 1 : ui1
// CHECK:             %[[GET_16:.*]] = coredsl.get @STRUCT_REG_x : ui32
// CHECK:             %[[GET_17:.*]] = coredsl.get @STRUCT_REG_y : ui32
// CHECK:             %[[ADD_0:.*]] = hwarith.add %[[GET_16]], %[[CONSTANT_4]] : (ui32, ui1) -> ui33
// CHECK:             %[[CAST_4:.*]] = coredsl.cast %[[ADD_0]] : ui33 to ui32
// CHECK:             coredsl.set @SCALAR_REG1 = %[[CAST_4]] : ui32
// CHECK:             coredsl.set @SCALAR_REG2 = %[[GET_17]] : ui32
// CHECK:             coredsl.end
// CHECK:           }
// CHECK:           coredsl.instruction @StructArrays {lil.enc_immediates = {{\[\[}}["%[[VAL_6:.*]]", 11, 0, 0, "imm"]], {{\[\[}}"%[[VAL_7:.*]]", 4, 0, 0, "rs1"]], {{\[\[}}"%[[VAL_8:.*]]", 4, 0, 0, "rd"]]]}(%[[VAL_6]] : ui12, %[[VAL_7]] : ui5, "010", %[[VAL_8]] : ui5, "0000011"){
// CHECK:             %[[CONSTANT_5:.*]] = hwarith.constant 10 : ui4
// CHECK:             %[[CAST_5:.*]] = coredsl.cast %[[VAL_7]] : ui5 to ui5
// CHECK:             %[[GET_18:.*]] = coredsl.get @NESTED_STRUCT_REG_notNested : si32
// CHECK:             %[[GET_19:.*]] = coredsl.get @NESTED_STRUCT_REG_vec_x : ui32
// CHECK:             %[[GET_20:.*]] = coredsl.get @NESTED_STRUCT_REG_vec_y : ui32
// CHECK:             coredsl.set @STRUCT_REGS_notNested[2] = %[[GET_18]] : si32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x[2] = %[[GET_19]] : ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y[2] = %[[GET_20]] : ui32
// CHECK:             %[[GET_21:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_5]] : ui5] : si32
// CHECK:             %[[GET_22:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_5]] : ui5] : ui32
// CHECK:             %[[GET_23:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_5]] : ui5] : ui32
// CHECK:             %[[CAST_6:.*]] = coredsl.cast %[[CONSTANT_5]] : ui4 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_5]] : ui5] = %[[CAST_6]] : si32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_5]] : ui5] = %[[GET_22]] : ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_5]] : ui5] = %[[GET_23]] : ui32
// CHECK:             coredsl.end
// CHECK:           }
// CHECK:         }
