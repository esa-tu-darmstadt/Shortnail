// RUN: shortnail-opt %s -coredsl-explode-struct-registers -canonicalize | shortnail-opt | FileCheck %s

coredsl.isax "StructRegisters" {
  coredsl.register local @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
  coredsl.register local @NESTED_STRUCT_REG : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
  coredsl.register local @TRIPLE_NESTED_REG : !hw.struct<internalStruct: !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>, intVal: ui32>
  coredsl.register local @SCALAR_REG1 : ui32
  coredsl.register local @SCALAR_REG2 : ui32
  coredsl.register local @STRUCT_REGS[32] : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
  coredsl.register local @OTHER_STRUCT_REGS[16] : !hw.struct<aValue: si64, aStruct: !hw.struct<vec: !hw.struct<x: ui32, y: ui32>, notNested: si32>>

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
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %21 = coredsl.get @NESTED_STRUCT_REG : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    coredsl.set @STRUCT_REGS[2] = %21 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %22 = hwarith.constant 10 : ui4
    %23 = coredsl.get @STRUCT_REGS[%rs1 : ui5] : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    %24 = coredsl.cast %22 : ui4 to si32
    %25 = hw.struct_inject %23["notNested"], %24 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    coredsl.set @STRUCT_REGS[%rs1 : ui5] = %25 : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
    // Ranged access for structs: structs get converted to integer
    %26 = coredsl.get @STRUCT_REGS[%rs1 : ui5, 0:4] : ui480
    coredsl.set @STRUCT_REGS[%rd : ui5, 0:4] = %26 : ui480
    // Nonzero offset
    %27 = coredsl.get @OTHER_STRUCT_REGS[%22 : ui4, 5:7] : ui480
    coredsl.set @OTHER_STRUCT_REGS[%22 : ui4, 1:3] = %27 : ui480
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
// CHECK:           coredsl.register local @OTHER_STRUCT_REGS_aValue[16]  : si64
// CHECK:           coredsl.register local @OTHER_STRUCT_REGS_aStruct_vec_x[16]  : ui32
// CHECK:           coredsl.register local @OTHER_STRUCT_REGS_aStruct_vec_y[16]  : ui32
// CHECK:           coredsl.register local @OTHER_STRUCT_REGS_aStruct_notNested[16]  : si32
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
// CHECK:             %[[CONSTANT_5:.*]] = hwarith.constant 7 : si4
// CHECK:             %[[CONSTANT_6:.*]] = hwarith.constant 6 : si4
// CHECK:             %[[CONSTANT_7:.*]] = hwarith.constant 5 : si4
// CHECK:             %[[CONSTANT_8:.*]] = hwarith.constant 4 : si4
// CHECK:             %[[CONSTANT_9:.*]] = hwarith.constant 3 : si3
// CHECK:             %[[CONSTANT_10:.*]] = hwarith.constant 2 : si3
// CHECK:             %[[CONSTANT_11:.*]] = hwarith.constant 1 : si2
// CHECK:             %[[CONSTANT_12:.*]] = hwarith.constant 10 : ui4
// CHECK:             %[[CAST_5:.*]] = coredsl.cast %[[VAL_7]] : ui5 to ui5
// CHECK:             %[[CAST_6:.*]] = coredsl.cast %[[VAL_8]] : ui5 to ui5
// CHECK:             %[[GET_18:.*]] = coredsl.get @NESTED_STRUCT_REG_notNested : si32
// CHECK:             %[[GET_19:.*]] = coredsl.get @NESTED_STRUCT_REG_vec_x : ui32
// CHECK:             %[[GET_20:.*]] = coredsl.get @NESTED_STRUCT_REG_vec_y : ui32
// CHECK:             coredsl.set @STRUCT_REGS_notNested[2] = %[[GET_18]] : si32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x[2] = %[[GET_19]] : ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y[2] = %[[GET_20]] : ui32
// CHECK:             %[[GET_21:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_5]] : ui5] : si32
// CHECK:             %[[GET_22:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_5]] : ui5] : ui32
// CHECK:             %[[GET_23:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_5]] : ui5] : ui32
// CHECK:             %[[CAST_7:.*]] = coredsl.cast %[[CONSTANT_12]] : ui4 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_5]] : ui5] = %[[CAST_7]] : si32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_5]] : ui5] = %[[GET_22]] : ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_5]] : ui5] = %[[GET_23]] : ui32
// CHECK:             %[[GET_24:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_5]] : ui5] : si32
// CHECK:             %[[CAST_8:.*]] = hwarith.cast %[[GET_24]] : (si32) -> i32
// CHECK:             %[[GET_25:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_5]] : ui5] : ui32
// CHECK:             %[[CAST_9:.*]] = hwarith.cast %[[GET_25]] : (ui32) -> i32
// CHECK:             %[[GET_26:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_5]] : ui5] : ui32
// CHECK:             %[[CAST_10:.*]] = hwarith.cast %[[GET_26]] : (ui32) -> i32
// CHECK:             %[[ADD_1:.*]] = hwarith.add %[[CAST_5]], %[[CONSTANT_11]] : (ui5, si2) -> si7
// CHECK:             %[[CAST_11:.*]] = hwarith.cast %[[ADD_1]] : (si7) -> ui5
// CHECK:             %[[GET_27:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_11]] : ui5] : si32
// CHECK:             %[[CAST_12:.*]] = hwarith.cast %[[GET_27]] : (si32) -> i32
// CHECK:             %[[GET_28:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_11]] : ui5] : ui32
// CHECK:             %[[CAST_13:.*]] = hwarith.cast %[[GET_28]] : (ui32) -> i32
// CHECK:             %[[GET_29:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_11]] : ui5] : ui32
// CHECK:             %[[CAST_14:.*]] = hwarith.cast %[[GET_29]] : (ui32) -> i32
// CHECK:             %[[ADD_2:.*]] = hwarith.add %[[CAST_5]], %[[CONSTANT_10]] : (ui5, si3) -> si7
// CHECK:             %[[CAST_15:.*]] = hwarith.cast %[[ADD_2]] : (si7) -> ui5
// CHECK:             %[[GET_30:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_15]] : ui5] : si32
// CHECK:             %[[CAST_16:.*]] = hwarith.cast %[[GET_30]] : (si32) -> i32
// CHECK:             %[[GET_31:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_15]] : ui5] : ui32
// CHECK:             %[[CAST_17:.*]] = hwarith.cast %[[GET_31]] : (ui32) -> i32
// CHECK:             %[[GET_32:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_15]] : ui5] : ui32
// CHECK:             %[[CAST_18:.*]] = hwarith.cast %[[GET_32]] : (ui32) -> i32
// CHECK:             %[[ADD_3:.*]] = hwarith.add %[[CAST_5]], %[[CONSTANT_9]] : (ui5, si3) -> si7
// CHECK:             %[[CAST_19:.*]] = hwarith.cast %[[ADD_3]] : (si7) -> ui5
// CHECK:             %[[GET_33:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_19]] : ui5] : si32
// CHECK:             %[[CAST_20:.*]] = hwarith.cast %[[GET_33]] : (si32) -> i32
// CHECK:             %[[GET_34:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_19]] : ui5] : ui32
// CHECK:             %[[CAST_21:.*]] = hwarith.cast %[[GET_34]] : (ui32) -> i32
// CHECK:             %[[GET_35:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_19]] : ui5] : ui32
// CHECK:             %[[CAST_22:.*]] = hwarith.cast %[[GET_35]] : (ui32) -> i32
// CHECK:             %[[ADD_4:.*]] = hwarith.add %[[CAST_5]], %[[CONSTANT_8]] : (ui5, si4) -> si7
// CHECK:             %[[CAST_23:.*]] = hwarith.cast %[[ADD_4]] : (si7) -> ui5
// CHECK:             %[[GET_36:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_23]] : ui5] : si32
// CHECK:             %[[CAST_24:.*]] = hwarith.cast %[[GET_36]] : (si32) -> i32
// CHECK:             %[[GET_37:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_23]] : ui5] : ui32
// CHECK:             %[[CAST_25:.*]] = hwarith.cast %[[GET_37]] : (ui32) -> i32
// CHECK:             %[[GET_38:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_23]] : ui5] : ui32
// CHECK:             %[[CAST_26:.*]] = hwarith.cast %[[GET_38]] : (ui32) -> i32
// CHECK:             %[[CONCAT_0:.*]] = comb.concat %[[CAST_8]], %[[CAST_9]], %[[CAST_10]], %[[CAST_12]], %[[CAST_13]], %[[CAST_14]], %[[CAST_16]], %[[CAST_17]], %[[CAST_18]], %[[CAST_20]], %[[CAST_21]], %[[CAST_22]], %[[CAST_24]], %[[CAST_25]], %[[CAST_26]] : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
// CHECK:             %[[CAST_27:.*]] = hwarith.cast %[[CONCAT_0]] : (i480) -> ui480
// CHECK:             %[[BITEXTRACT_0:.*]] = coredsl.bitextract %[[CAST_27]][0:31] : (ui480) -> ui32
// CHECK:             %[[CAST_28:.*]] = coredsl.cast %[[BITEXTRACT_0]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_6]] : ui5] = %[[CAST_28]] : si32
// CHECK:             %[[BITEXTRACT_1:.*]] = coredsl.bitextract %[[CAST_27]][32:63] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_6]] : ui5] = %[[BITEXTRACT_1]] : ui32
// CHECK:             %[[BITEXTRACT_2:.*]] = coredsl.bitextract %[[CAST_27]][64:95] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_6]] : ui5] = %[[BITEXTRACT_2]] : ui32
// CHECK:             %[[ADD_5:.*]] = hwarith.add %[[CAST_6]], %[[CONSTANT_11]] : (ui5, si2) -> si7
// CHECK:             %[[CAST_29:.*]] = hwarith.cast %[[ADD_5]] : (si7) -> ui5
// CHECK:             %[[BITEXTRACT_3:.*]] = coredsl.bitextract %[[CAST_27]][96:127] : (ui480) -> ui32
// CHECK:             %[[CAST_30:.*]] = coredsl.cast %[[BITEXTRACT_3]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_29]] : ui5] = %[[CAST_30]] : si32
// CHECK:             %[[BITEXTRACT_4:.*]] = coredsl.bitextract %[[CAST_27]][128:159] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_29]] : ui5] = %[[BITEXTRACT_4]] : ui32
// CHECK:             %[[BITEXTRACT_5:.*]] = coredsl.bitextract %[[CAST_27]][160:191] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_29]] : ui5] = %[[BITEXTRACT_5]] : ui32
// CHECK:             %[[ADD_6:.*]] = hwarith.add %[[CAST_6]], %[[CONSTANT_10]] : (ui5, si3) -> si7
// CHECK:             %[[CAST_31:.*]] = hwarith.cast %[[ADD_6]] : (si7) -> ui5
// CHECK:             %[[BITEXTRACT_6:.*]] = coredsl.bitextract %[[CAST_27]][192:223] : (ui480) -> ui32
// CHECK:             %[[CAST_32:.*]] = coredsl.cast %[[BITEXTRACT_6]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_31]] : ui5] = %[[CAST_32]] : si32
// CHECK:             %[[BITEXTRACT_7:.*]] = coredsl.bitextract %[[CAST_27]][224:255] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_31]] : ui5] = %[[BITEXTRACT_7]] : ui32
// CHECK:             %[[BITEXTRACT_8:.*]] = coredsl.bitextract %[[CAST_27]][256:287] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_31]] : ui5] = %[[BITEXTRACT_8]] : ui32
// CHECK:             %[[ADD_7:.*]] = hwarith.add %[[CAST_6]], %[[CONSTANT_9]] : (ui5, si3) -> si7
// CHECK:             %[[CAST_33:.*]] = hwarith.cast %[[ADD_7]] : (si7) -> ui5
// CHECK:             %[[BITEXTRACT_9:.*]] = coredsl.bitextract %[[CAST_27]][288:319] : (ui480) -> ui32
// CHECK:             %[[CAST_34:.*]] = coredsl.cast %[[BITEXTRACT_9]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_33]] : ui5] = %[[CAST_34]] : si32
// CHECK:             %[[BITEXTRACT_10:.*]] = coredsl.bitextract %[[CAST_27]][320:351] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_33]] : ui5] = %[[BITEXTRACT_10]] : ui32
// CHECK:             %[[BITEXTRACT_11:.*]] = coredsl.bitextract %[[CAST_27]][352:383] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_33]] : ui5] = %[[BITEXTRACT_11]] : ui32
// CHECK:             %[[ADD_8:.*]] = hwarith.add %[[CAST_6]], %[[CONSTANT_8]] : (ui5, si4) -> si7
// CHECK:             %[[CAST_35:.*]] = hwarith.cast %[[ADD_8]] : (si7) -> ui5
// CHECK:             %[[BITEXTRACT_12:.*]] = coredsl.bitextract %[[CAST_27]][384:415] : (ui480) -> ui32
// CHECK:             %[[CAST_36:.*]] = coredsl.cast %[[BITEXTRACT_12]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_35]] : ui5] = %[[CAST_36]] : si32
// CHECK:             %[[BITEXTRACT_13:.*]] = coredsl.bitextract %[[CAST_27]][416:447] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_35]] : ui5] = %[[BITEXTRACT_13]] : ui32
// CHECK:             %[[BITEXTRACT_14:.*]] = coredsl.bitextract %[[CAST_27]][448:479] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_35]] : ui5] = %[[BITEXTRACT_14]] : ui32
// CHECK:             %[[ADD_9:.*]] = hwarith.add %[[CONSTANT_12]], %[[CONSTANT_7]] : (ui4, si4) -> si6
// CHECK:             %[[CAST_37:.*]] = hwarith.cast %[[ADD_9]] : (si6) -> ui4
// CHECK:             %[[GET_39:.*]] = coredsl.get @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_37]] : ui4] : si64
// CHECK:             %[[CAST_38:.*]] = hwarith.cast %[[GET_39]] : (si64) -> i64
// CHECK:             %[[GET_40:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_37]] : ui4] : ui32
// CHECK:             %[[CAST_39:.*]] = hwarith.cast %[[GET_40]] : (ui32) -> i32
// CHECK:             %[[GET_41:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_37]] : ui4] : ui32
// CHECK:             %[[CAST_40:.*]] = hwarith.cast %[[GET_41]] : (ui32) -> i32
// CHECK:             %[[GET_42:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_37]] : ui4] : si32
// CHECK:             %[[CAST_41:.*]] = hwarith.cast %[[GET_42]] : (si32) -> i32
// CHECK:             %[[ADD_10:.*]] = hwarith.add %[[CONSTANT_12]], %[[CONSTANT_6]] : (ui4, si4) -> si6
// CHECK:             %[[CAST_42:.*]] = hwarith.cast %[[ADD_10]] : (si6) -> ui4
// CHECK:             %[[GET_43:.*]] = coredsl.get @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_42]] : ui4] : si64
// CHECK:             %[[CAST_43:.*]] = hwarith.cast %[[GET_43]] : (si64) -> i64
// CHECK:             %[[GET_44:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_42]] : ui4] : ui32
// CHECK:             %[[CAST_44:.*]] = hwarith.cast %[[GET_44]] : (ui32) -> i32
// CHECK:             %[[GET_45:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_42]] : ui4] : ui32
// CHECK:             %[[CAST_45:.*]] = hwarith.cast %[[GET_45]] : (ui32) -> i32
// CHECK:             %[[GET_46:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_42]] : ui4] : si32
// CHECK:             %[[CAST_46:.*]] = hwarith.cast %[[GET_46]] : (si32) -> i32
// CHECK:             %[[ADD_11:.*]] = hwarith.add %[[CONSTANT_12]], %[[CONSTANT_5]] : (ui4, si4) -> si6
// CHECK:             %[[CAST_47:.*]] = hwarith.cast %[[ADD_11]] : (si6) -> ui4
// CHECK:             %[[GET_47:.*]] = coredsl.get @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_47]] : ui4] : si64
// CHECK:             %[[CAST_48:.*]] = hwarith.cast %[[GET_47]] : (si64) -> i64
// CHECK:             %[[GET_48:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_47]] : ui4] : ui32
// CHECK:             %[[CAST_49:.*]] = hwarith.cast %[[GET_48]] : (ui32) -> i32
// CHECK:             %[[GET_49:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_47]] : ui4] : ui32
// CHECK:             %[[CAST_50:.*]] = hwarith.cast %[[GET_49]] : (ui32) -> i32
// CHECK:             %[[GET_50:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_47]] : ui4] : si32
// CHECK:             %[[CAST_51:.*]] = hwarith.cast %[[GET_50]] : (si32) -> i32
// CHECK:             %[[CONCAT_1:.*]] = comb.concat %[[CAST_38]], %[[CAST_39]], %[[CAST_40]], %[[CAST_41]], %[[CAST_43]], %[[CAST_44]], %[[CAST_45]], %[[CAST_46]], %[[CAST_48]], %[[CAST_49]], %[[CAST_50]], %[[CAST_51]] : i64, i32, i32, i32, i64, i32, i32, i32, i64, i32, i32, i32
// CHECK:             %[[CAST_52:.*]] = hwarith.cast %[[CONCAT_1]] : (i480) -> ui480
// CHECK:             %[[ADD_12:.*]] = hwarith.add %[[CONSTANT_12]], %[[CONSTANT_11]] : (ui4, si2) -> si6
// CHECK:             %[[CAST_53:.*]] = hwarith.cast %[[ADD_12]] : (si6) -> ui4
// CHECK:             %[[BITEXTRACT_15:.*]] = coredsl.bitextract %[[CAST_52]][0:63] : (ui480) -> ui64
// CHECK:             %[[CAST_54:.*]] = coredsl.cast %[[BITEXTRACT_15]] : ui64 to si64
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_53]] : ui4] = %[[CAST_54]] : si64
// CHECK:             %[[BITEXTRACT_16:.*]] = coredsl.bitextract %[[CAST_52]][64:95] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_53]] : ui4] = %[[BITEXTRACT_16]] : ui32
// CHECK:             %[[BITEXTRACT_17:.*]] = coredsl.bitextract %[[CAST_52]][96:127] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_53]] : ui4] = %[[BITEXTRACT_17]] : ui32
// CHECK:             %[[BITEXTRACT_18:.*]] = coredsl.bitextract %[[CAST_52]][128:159] : (ui480) -> ui32
// CHECK:             %[[CAST_55:.*]] = coredsl.cast %[[BITEXTRACT_18]] : ui32 to si32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_53]] : ui4] = %[[CAST_55]] : si32
// CHECK:             %[[ADD_13:.*]] = hwarith.add %[[CONSTANT_12]], %[[CONSTANT_10]] : (ui4, si3) -> si6
// CHECK:             %[[CAST_56:.*]] = hwarith.cast %[[ADD_13]] : (si6) -> ui4
// CHECK:             %[[BITEXTRACT_19:.*]] = coredsl.bitextract %[[CAST_52]][160:223] : (ui480) -> ui64
// CHECK:             %[[CAST_57:.*]] = coredsl.cast %[[BITEXTRACT_19]] : ui64 to si64
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_56]] : ui4] = %[[CAST_57]] : si64
// CHECK:             %[[BITEXTRACT_20:.*]] = coredsl.bitextract %[[CAST_52]][224:255] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_56]] : ui4] = %[[BITEXTRACT_20]] : ui32
// CHECK:             %[[BITEXTRACT_21:.*]] = coredsl.bitextract %[[CAST_52]][256:287] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_56]] : ui4] = %[[BITEXTRACT_21]] : ui32
// CHECK:             %[[BITEXTRACT_22:.*]] = coredsl.bitextract %[[CAST_52]][288:319] : (ui480) -> ui32
// CHECK:             %[[CAST_58:.*]] = coredsl.cast %[[BITEXTRACT_22]] : ui32 to si32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_56]] : ui4] = %[[CAST_58]] : si32
// CHECK:             %[[ADD_14:.*]] = hwarith.add %[[CONSTANT_12]], %[[CONSTANT_9]] : (ui4, si3) -> si6
// CHECK:             %[[CAST_59:.*]] = hwarith.cast %[[ADD_14]] : (si6) -> ui4
// CHECK:             %[[BITEXTRACT_23:.*]] = coredsl.bitextract %[[CAST_52]][320:383] : (ui480) -> ui64
// CHECK:             %[[CAST_60:.*]] = coredsl.cast %[[BITEXTRACT_23]] : ui64 to si64
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_59]] : ui4] = %[[CAST_60]] : si64
// CHECK:             %[[BITEXTRACT_24:.*]] = coredsl.bitextract %[[CAST_52]][384:415] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_59]] : ui4] = %[[BITEXTRACT_24]] : ui32
// CHECK:             %[[BITEXTRACT_25:.*]] = coredsl.bitextract %[[CAST_52]][416:447] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_59]] : ui4] = %[[BITEXTRACT_25]] : ui32
// CHECK:             %[[BITEXTRACT_26:.*]] = coredsl.bitextract %[[CAST_52]][448:479] : (ui480) -> ui32
// CHECK:             %[[CAST_61:.*]] = coredsl.cast %[[BITEXTRACT_26]] : ui32 to si32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_59]] : ui4] = %[[CAST_61]] : si32
// CHECK:             coredsl.end
// CHECK:           }
// CHECK:         }
