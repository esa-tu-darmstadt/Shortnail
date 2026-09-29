// RUN: shortnail-opt %s -coredsl-explode-struct-registers -canonicalize | shortnail-opt | FileCheck %s

coredsl.isax "StructRegisters" {
  coredsl.register local @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
  coredsl.register local @OTHER_STRUCT_REG : !hw.struct<x: ui32, y: ui32>
  coredsl.register local @NESTED_STRUCT_REG : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
  coredsl.register local @TRIPLE_NESTED_REG : !hw.struct<internalStruct: !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>, intVal: ui32>
  coredsl.register local @SCALAR_REG1 : ui32
  coredsl.register local @SCALAR_REG2 : ui32
  coredsl.register local @STRUCT_REGS[32] : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
  coredsl.register local @OTHER_STRUCT_REGS[16] : !hw.struct<aValue: si64, aStruct: !hw.struct<vec: !hw.struct<x: ui32, y: ui32>, notNested: si32>>

  func.func @testWritingBlockArgument(%v : !hw.struct<x: ui32, y: ui32>) {
    coredsl.set @STRUCT_REG = %v : !hw.struct<x: ui32, y: ui32>
    return
  }

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
    // No base offset
    %28 = coredsl.get @STRUCT_REGS[5:7] : ui288
    coredsl.set @STRUCT_REGS[6:8] = %28 : ui288
    // Narrow base index: Previously, the index to read part of the range was
    // emitted without a cast, which ignored the fact that the index may be
    // signed, causing invalid mlir to be emitted
    %29 = coredsl.get @STRUCT_REGS[%22 : ui4, 0:1] : ui192
    coredsl.set @STRUCT_REGS[%22 : ui4, 0:1] = %29 : ui192
    // ranged access with to > from (big-endian)
    %30 = coredsl.get @STRUCT_REGS[%22 : ui4, 5:3] : ui288
    coredsl.set @STRUCT_REGS[%22 : ui4, 5:3] = %30 : ui288
    // Special case: no base index with zero offset
    %31 = coredsl.get @STRUCT_REGS[0:1] : ui192
    coredsl.set @STRUCT_REGS[0:1] = %31 : ui192
    coredsl.end
  }
  coredsl.instruction @MultipleReturnValues("0000000", %rs1 : ui5, "00000000000000000000") {
    %c = hwarith.constant 1 : ui1
    %b = coredsl.cast %c : ui1 to i1
    %a = coredsl.get @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
    %q = coredsl.get @OTHER_STRUCT_REG : !hw.struct<x: ui32, y: ui32>
    %r:2 = scf.if %b -> (!hw.struct<x: ui32, y: ui32>, !hw.struct<x: ui32, y: ui32>) {
      scf.yield %a, %q : !hw.struct<x: ui32, y: ui32>, !hw.struct<x: ui32, y: ui32>
    } else {
      scf.yield %q, %a : !hw.struct<x: ui32, y: ui32>, !hw.struct<x: ui32, y: ui32>
    }
    coredsl.set @STRUCT_REG = %r#1 : !hw.struct<x: ui32, y: ui32>
    coredsl.end
  }
}

// CHECK-LABEL:   coredsl.isax "StructRegisters" {
// CHECK:           coredsl.register local @STRUCT_REG_x  : ui32
// CHECK:           coredsl.register local @STRUCT_REG_y  : ui32
// CHECK:           coredsl.register local @OTHER_STRUCT_REG_x  : ui32
// CHECK:           coredsl.register local @OTHER_STRUCT_REG_y  : ui32
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
// CHECK:           func.func @testWritingBlockArgument(%[[VAL_0:.*]]: !hw.struct<x: ui32, y: ui32>) {
// CHECK:             %[[STRUCT_EXTRACT_0:.*]] = hw.struct_extract %[[VAL_0]]["x"] : !hw.struct<x: ui32, y: ui32>
// CHECK:             coredsl.set @STRUCT_REG_x = %[[STRUCT_EXTRACT_0]] : ui32
// CHECK:             %[[STRUCT_EXTRACT_1:.*]] = hw.struct_extract %[[VAL_0]]["y"] : !hw.struct<x: ui32, y: ui32>
// CHECK:             coredsl.set @STRUCT_REG_y = %[[STRUCT_EXTRACT_1]] : ui32
// CHECK:             return
// CHECK:           }
// CHECK:           coredsl.instruction @StructRegDirectStore {lil.enc_immediates = {{\[\[}}["%[[VAL_1:.*]]", 11, 0, 0, "imm"]], {{\[\[}}"%[[VAL_2:.*]]", 4, 0, 0, "rs1"]], {{\[\[}}"%[[VAL_3:.*]]", 4, 0, 0, "rd"]]]}(%[[VAL_1]] : ui12, %[[VAL_2]] : ui5, "010", %[[VAL_3]] : ui5, "0000011"){
// CHECK:             %[[CONSTANT_0:.*]] = hwarith.constant -1 : si32
// CHECK:             %[[CONSTANT_1:.*]] = hwarith.constant 0 : ui1
// CHECK:             %[[CONSTANT_2:.*]] = hwarith.constant 255 : ui8
// CHECK:             %[[CONSTANT_3:.*]] = hwarith.constant 7 : ui3
// CHECK:             %[[CAST_0:.*]] = coredsl.cast %[[VAL_2]] : ui5 to ui5
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
// CHECK:           coredsl.instruction @TransferStructToScalarReg {lil.enc_immediates = {{\[\[}}["%[[VAL_4:.*]]", 11, 0, 0, "imm"]], {{\[\[}}"%[[VAL_5:.*]]", 4, 0, 0, "rs1"]], {{\[\[}}"%[[VAL_6:.*]]", 4, 0, 0, "rd"]]]}(%[[VAL_4]] : ui12, %[[VAL_5]] : ui5, "010", %[[VAL_6]] : ui5, "0000011"){
// CHECK:             %[[CONSTANT_4:.*]] = hwarith.constant 1 : ui1
// CHECK:             %[[GET_16:.*]] = coredsl.get @STRUCT_REG_x : ui32
// CHECK:             %[[GET_17:.*]] = coredsl.get @STRUCT_REG_y : ui32
// CHECK:             %[[ADD_0:.*]] = hwarith.add %[[GET_16]], %[[CONSTANT_4]] : (ui32, ui1) -> ui33
// CHECK:             %[[CAST_4:.*]] = coredsl.cast %[[ADD_0]] : ui33 to ui32
// CHECK:             coredsl.set @SCALAR_REG1 = %[[CAST_4]] : ui32
// CHECK:             coredsl.set @SCALAR_REG2 = %[[GET_17]] : ui32
// CHECK:             coredsl.end
// CHECK:           }
// CHECK:           coredsl.instruction @StructArrays {lil.enc_immediates = {{\[\[}}["%[[VAL_7:.*]]", 11, 0, 0, "imm"]], {{\[\[}}"%[[VAL_8:.*]]", 4, 0, 0, "rs1"]], {{\[\[}}"%[[VAL_9:.*]]", 4, 0, 0, "rd"]]]}(%[[VAL_7]] : ui12, %[[VAL_8]] : ui5, "010", %[[VAL_9]] : ui5, "0000011"){
// CHECK:             %[[CONSTANT_5:.*]] = hwarith.constant 0 : si1
// CHECK:             %[[CONSTANT_6:.*]] = hwarith.constant 8 : si5
// CHECK:             %[[CONSTANT_7:.*]] = hwarith.constant 7 : si4
// CHECK:             %[[CONSTANT_8:.*]] = hwarith.constant 6 : si4
// CHECK:             %[[CONSTANT_9:.*]] = hwarith.constant 5 : si4
// CHECK:             %[[CONSTANT_10:.*]] = hwarith.constant 4 : si4
// CHECK:             %[[CONSTANT_11:.*]] = hwarith.constant 3 : si3
// CHECK:             %[[CONSTANT_12:.*]] = hwarith.constant 2 : si3
// CHECK:             %[[CONSTANT_13:.*]] = hwarith.constant 1 : si2
// CHECK:             %[[CONSTANT_14:.*]] = hwarith.constant 10 : ui4
// CHECK:             %[[CAST_5:.*]] = coredsl.cast %[[VAL_8]] : ui5 to ui5
// CHECK:             %[[CAST_6:.*]] = coredsl.cast %[[VAL_9]] : ui5 to ui5
// CHECK:             %[[GET_18:.*]] = coredsl.get @NESTED_STRUCT_REG_notNested : si32
// CHECK:             %[[GET_19:.*]] = coredsl.get @NESTED_STRUCT_REG_vec_x : ui32
// CHECK:             %[[GET_20:.*]] = coredsl.get @NESTED_STRUCT_REG_vec_y : ui32
// CHECK:             coredsl.set @STRUCT_REGS_notNested[2] = %[[GET_18]] : si32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x[2] = %[[GET_19]] : ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y[2] = %[[GET_20]] : ui32
// CHECK:             %[[GET_21:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_5]] : ui5] : si32
// CHECK:             %[[GET_22:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_5]] : ui5] : ui32
// CHECK:             %[[GET_23:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_5]] : ui5] : ui32
// CHECK:             %[[CAST_7:.*]] = coredsl.cast %[[CONSTANT_14]] : ui4 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_5]] : ui5] = %[[CAST_7]] : si32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_5]] : ui5] = %[[GET_22]] : ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_5]] : ui5] = %[[GET_23]] : ui32
// CHECK:             %[[GET_24:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_5]] : ui5] : si32
// CHECK:             %[[GET_25:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_5]] : ui5] : ui32
// CHECK:             %[[GET_26:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_5]] : ui5] : ui32
// CHECK:             %[[ADD_1:.*]] = hwarith.add %[[CAST_5]], %[[CONSTANT_13]] : (ui5, si2) -> si7
// CHECK:             %[[CAST_8:.*]] = hwarith.cast %[[ADD_1]] : (si7) -> ui5
// CHECK:             %[[GET_27:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_8]] : ui5] : si32
// CHECK:             %[[GET_28:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_8]] : ui5] : ui32
// CHECK:             %[[GET_29:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_8]] : ui5] : ui32
// CHECK:             %[[ADD_2:.*]] = hwarith.add %[[CAST_5]], %[[CONSTANT_12]] : (ui5, si3) -> si7
// CHECK:             %[[CAST_9:.*]] = hwarith.cast %[[ADD_2]] : (si7) -> ui5
// CHECK:             %[[GET_30:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_9]] : ui5] : si32
// CHECK:             %[[GET_31:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_9]] : ui5] : ui32
// CHECK:             %[[GET_32:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_9]] : ui5] : ui32
// CHECK:             %[[ADD_3:.*]] = hwarith.add %[[CAST_5]], %[[CONSTANT_11]] : (ui5, si3) -> si7
// CHECK:             %[[CAST_10:.*]] = hwarith.cast %[[ADD_3]] : (si7) -> ui5
// CHECK:             %[[GET_33:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_10]] : ui5] : si32
// CHECK:             %[[GET_34:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_10]] : ui5] : ui32
// CHECK:             %[[GET_35:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_10]] : ui5] : ui32
// CHECK:             %[[ADD_4:.*]] = hwarith.add %[[CAST_5]], %[[CONSTANT_10]] : (ui5, si4) -> si7
// CHECK:             %[[CAST_11:.*]] = hwarith.cast %[[ADD_4]] : (si7) -> ui5
// CHECK:             %[[GET_36:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_11]] : ui5] : si32
// CHECK:             %[[GET_37:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_11]] : ui5] : ui32
// CHECK:             %[[GET_38:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_11]] : ui5] : ui32
// CHECK:             %[[CONCAT_0:.*]] = coredsl.concat %[[GET_24]], %[[GET_25]] : si32, ui32
// CHECK:             %[[CONCAT_1:.*]] = coredsl.concat %[[CONCAT_0]], %[[GET_26]] : ui64, ui32
// CHECK:             %[[CONCAT_2:.*]] = coredsl.concat %[[CONCAT_1]], %[[GET_27]] : ui96, si32
// CHECK:             %[[CONCAT_3:.*]] = coredsl.concat %[[CONCAT_2]], %[[GET_28]] : ui128, ui32
// CHECK:             %[[CONCAT_4:.*]] = coredsl.concat %[[CONCAT_3]], %[[GET_29]] : ui160, ui32
// CHECK:             %[[CONCAT_5:.*]] = coredsl.concat %[[CONCAT_4]], %[[GET_30]] : ui192, si32
// CHECK:             %[[CONCAT_6:.*]] = coredsl.concat %[[CONCAT_5]], %[[GET_31]] : ui224, ui32
// CHECK:             %[[CONCAT_7:.*]] = coredsl.concat %[[CONCAT_6]], %[[GET_32]] : ui256, ui32
// CHECK:             %[[CONCAT_8:.*]] = coredsl.concat %[[CONCAT_7]], %[[GET_33]] : ui288, si32
// CHECK:             %[[CONCAT_9:.*]] = coredsl.concat %[[CONCAT_8]], %[[GET_34]] : ui320, ui32
// CHECK:             %[[CONCAT_10:.*]] = coredsl.concat %[[CONCAT_9]], %[[GET_35]] : ui352, ui32
// CHECK:             %[[CONCAT_11:.*]] = coredsl.concat %[[CONCAT_10]], %[[GET_36]] : ui384, si32
// CHECK:             %[[CONCAT_12:.*]] = coredsl.concat %[[CONCAT_11]], %[[GET_37]] : ui416, ui32
// CHECK:             %[[CONCAT_13:.*]] = coredsl.concat %[[CONCAT_12]], %[[GET_38]] : ui448, ui32
// CHECK:             %[[CAST_12:.*]] = coredsl.cast %[[CONCAT_13]] : ui480 to ui32
// CHECK:             %[[CAST_13:.*]] = coredsl.cast %[[CAST_12]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_6]] : ui5] = %[[CAST_13]] : si32
// CHECK:             %[[BITEXTRACT_0:.*]] = coredsl.bitextract %[[CONCAT_13]][63:32] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_6]] : ui5] = %[[BITEXTRACT_0]] : ui32
// CHECK:             %[[BITEXTRACT_1:.*]] = coredsl.bitextract %[[CONCAT_13]][95:64] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_6]] : ui5] = %[[BITEXTRACT_1]] : ui32
// CHECK:             %[[ADD_5:.*]] = hwarith.add %[[CAST_6]], %[[CONSTANT_13]] : (ui5, si2) -> si7
// CHECK:             %[[CAST_14:.*]] = hwarith.cast %[[ADD_5]] : (si7) -> ui5
// CHECK:             %[[BITEXTRACT_2:.*]] = coredsl.bitextract %[[CONCAT_13]][127:96] : (ui480) -> ui32
// CHECK:             %[[CAST_15:.*]] = coredsl.cast %[[BITEXTRACT_2]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_14]] : ui5] = %[[CAST_15]] : si32
// CHECK:             %[[BITEXTRACT_3:.*]] = coredsl.bitextract %[[CONCAT_13]][159:128] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_14]] : ui5] = %[[BITEXTRACT_3]] : ui32
// CHECK:             %[[BITEXTRACT_4:.*]] = coredsl.bitextract %[[CONCAT_13]][191:160] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_14]] : ui5] = %[[BITEXTRACT_4]] : ui32
// CHECK:             %[[ADD_6:.*]] = hwarith.add %[[CAST_6]], %[[CONSTANT_12]] : (ui5, si3) -> si7
// CHECK:             %[[CAST_16:.*]] = hwarith.cast %[[ADD_6]] : (si7) -> ui5
// CHECK:             %[[BITEXTRACT_5:.*]] = coredsl.bitextract %[[CONCAT_13]][223:192] : (ui480) -> ui32
// CHECK:             %[[CAST_17:.*]] = coredsl.cast %[[BITEXTRACT_5]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_16]] : ui5] = %[[CAST_17]] : si32
// CHECK:             %[[BITEXTRACT_6:.*]] = coredsl.bitextract %[[CONCAT_13]][255:224] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_16]] : ui5] = %[[BITEXTRACT_6]] : ui32
// CHECK:             %[[BITEXTRACT_7:.*]] = coredsl.bitextract %[[CONCAT_13]][287:256] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_16]] : ui5] = %[[BITEXTRACT_7]] : ui32
// CHECK:             %[[ADD_7:.*]] = hwarith.add %[[CAST_6]], %[[CONSTANT_11]] : (ui5, si3) -> si7
// CHECK:             %[[CAST_18:.*]] = hwarith.cast %[[ADD_7]] : (si7) -> ui5
// CHECK:             %[[BITEXTRACT_8:.*]] = coredsl.bitextract %[[CONCAT_13]][319:288] : (ui480) -> ui32
// CHECK:             %[[CAST_19:.*]] = coredsl.cast %[[BITEXTRACT_8]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_18]] : ui5] = %[[CAST_19]] : si32
// CHECK:             %[[BITEXTRACT_9:.*]] = coredsl.bitextract %[[CONCAT_13]][351:320] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_18]] : ui5] = %[[BITEXTRACT_9]] : ui32
// CHECK:             %[[BITEXTRACT_10:.*]] = coredsl.bitextract %[[CONCAT_13]][383:352] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_18]] : ui5] = %[[BITEXTRACT_10]] : ui32
// CHECK:             %[[ADD_8:.*]] = hwarith.add %[[CAST_6]], %[[CONSTANT_10]] : (ui5, si4) -> si7
// CHECK:             %[[CAST_20:.*]] = hwarith.cast %[[ADD_8]] : (si7) -> ui5
// CHECK:             %[[BITEXTRACT_11:.*]] = coredsl.bitextract %[[CONCAT_13]][415:384] : (ui480) -> ui32
// CHECK:             %[[CAST_21:.*]] = coredsl.cast %[[BITEXTRACT_11]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_20]] : ui5] = %[[CAST_21]] : si32
// CHECK:             %[[BITEXTRACT_12:.*]] = coredsl.bitextract %[[CONCAT_13]][447:416] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_20]] : ui5] = %[[BITEXTRACT_12]] : ui32
// CHECK:             %[[BITEXTRACT_13:.*]] = coredsl.bitextract %[[CONCAT_13]][479:448] : (ui480) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_20]] : ui5] = %[[BITEXTRACT_13]] : ui32
// CHECK:             %[[ADD_9:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_9]] : (ui4, si4) -> si6
// CHECK:             %[[CAST_22:.*]] = hwarith.cast %[[ADD_9]] : (si6) -> ui4
// CHECK:             %[[GET_39:.*]] = coredsl.get @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_22]] : ui4] : si64
// CHECK:             %[[GET_40:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_22]] : ui4] : ui32
// CHECK:             %[[GET_41:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_22]] : ui4] : ui32
// CHECK:             %[[GET_42:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_22]] : ui4] : si32
// CHECK:             %[[ADD_10:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_8]] : (ui4, si4) -> si6
// CHECK:             %[[CAST_23:.*]] = hwarith.cast %[[ADD_10]] : (si6) -> ui4
// CHECK:             %[[GET_43:.*]] = coredsl.get @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_23]] : ui4] : si64
// CHECK:             %[[GET_44:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_23]] : ui4] : ui32
// CHECK:             %[[GET_45:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_23]] : ui4] : ui32
// CHECK:             %[[GET_46:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_23]] : ui4] : si32
// CHECK:             %[[ADD_11:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_7]] : (ui4, si4) -> si6
// CHECK:             %[[CAST_24:.*]] = hwarith.cast %[[ADD_11]] : (si6) -> ui4
// CHECK:             %[[GET_47:.*]] = coredsl.get @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_24]] : ui4] : si64
// CHECK:             %[[GET_48:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_24]] : ui4] : ui32
// CHECK:             %[[GET_49:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_24]] : ui4] : ui32
// CHECK:             %[[GET_50:.*]] = coredsl.get @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_24]] : ui4] : si32
// CHECK:             %[[CONCAT_14:.*]] = coredsl.concat %[[GET_39]], %[[GET_40]] : si64, ui32
// CHECK:             %[[CONCAT_15:.*]] = coredsl.concat %[[CONCAT_14]], %[[GET_41]] : ui96, ui32
// CHECK:             %[[CONCAT_16:.*]] = coredsl.concat %[[CONCAT_15]], %[[GET_42]] : ui128, si32
// CHECK:             %[[CONCAT_17:.*]] = coredsl.concat %[[CONCAT_16]], %[[GET_43]] : ui160, si64
// CHECK:             %[[CONCAT_18:.*]] = coredsl.concat %[[CONCAT_17]], %[[GET_44]] : ui224, ui32
// CHECK:             %[[CONCAT_19:.*]] = coredsl.concat %[[CONCAT_18]], %[[GET_45]] : ui256, ui32
// CHECK:             %[[CONCAT_20:.*]] = coredsl.concat %[[CONCAT_19]], %[[GET_46]] : ui288, si32
// CHECK:             %[[CONCAT_21:.*]] = coredsl.concat %[[CONCAT_20]], %[[GET_47]] : ui320, si64
// CHECK:             %[[CONCAT_22:.*]] = coredsl.concat %[[CONCAT_21]], %[[GET_48]] : ui384, ui32
// CHECK:             %[[CONCAT_23:.*]] = coredsl.concat %[[CONCAT_22]], %[[GET_49]] : ui416, ui32
// CHECK:             %[[CONCAT_24:.*]] = coredsl.concat %[[CONCAT_23]], %[[GET_50]] : ui448, si32
// CHECK:             %[[ADD_12:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_13]] : (ui4, si2) -> si6
// CHECK:             %[[CAST_25:.*]] = hwarith.cast %[[ADD_12]] : (si6) -> ui4
// CHECK:             %[[CAST_26:.*]] = coredsl.cast %[[CONCAT_24]] : ui480 to ui64
// CHECK:             %[[CAST_27:.*]] = coredsl.cast %[[CAST_26]] : ui64 to si64
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_25]] : ui4] = %[[CAST_27]] : si64
// CHECK:             %[[BITEXTRACT_14:.*]] = coredsl.bitextract %[[CONCAT_24]][95:64] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_25]] : ui4] = %[[BITEXTRACT_14]] : ui32
// CHECK:             %[[BITEXTRACT_15:.*]] = coredsl.bitextract %[[CONCAT_24]][127:96] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_25]] : ui4] = %[[BITEXTRACT_15]] : ui32
// CHECK:             %[[BITEXTRACT_16:.*]] = coredsl.bitextract %[[CONCAT_24]][159:128] : (ui480) -> ui32
// CHECK:             %[[CAST_28:.*]] = coredsl.cast %[[BITEXTRACT_16]] : ui32 to si32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_25]] : ui4] = %[[CAST_28]] : si32
// CHECK:             %[[ADD_13:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_12]] : (ui4, si3) -> si6
// CHECK:             %[[CAST_29:.*]] = hwarith.cast %[[ADD_13]] : (si6) -> ui4
// CHECK:             %[[BITEXTRACT_17:.*]] = coredsl.bitextract %[[CONCAT_24]][223:160] : (ui480) -> ui64
// CHECK:             %[[CAST_30:.*]] = coredsl.cast %[[BITEXTRACT_17]] : ui64 to si64
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_29]] : ui4] = %[[CAST_30]] : si64
// CHECK:             %[[BITEXTRACT_18:.*]] = coredsl.bitextract %[[CONCAT_24]][255:224] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_29]] : ui4] = %[[BITEXTRACT_18]] : ui32
// CHECK:             %[[BITEXTRACT_19:.*]] = coredsl.bitextract %[[CONCAT_24]][287:256] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_29]] : ui4] = %[[BITEXTRACT_19]] : ui32
// CHECK:             %[[BITEXTRACT_20:.*]] = coredsl.bitextract %[[CONCAT_24]][319:288] : (ui480) -> ui32
// CHECK:             %[[CAST_31:.*]] = coredsl.cast %[[BITEXTRACT_20]] : ui32 to si32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_29]] : ui4] = %[[CAST_31]] : si32
// CHECK:             %[[ADD_14:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_11]] : (ui4, si3) -> si6
// CHECK:             %[[CAST_32:.*]] = hwarith.cast %[[ADD_14]] : (si6) -> ui4
// CHECK:             %[[BITEXTRACT_21:.*]] = coredsl.bitextract %[[CONCAT_24]][383:320] : (ui480) -> ui64
// CHECK:             %[[CAST_33:.*]] = coredsl.cast %[[BITEXTRACT_21]] : ui64 to si64
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aValue{{\[}}%[[CAST_32]] : ui4] = %[[CAST_33]] : si64
// CHECK:             %[[BITEXTRACT_22:.*]] = coredsl.bitextract %[[CONCAT_24]][415:384] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_x{{\[}}%[[CAST_32]] : ui4] = %[[BITEXTRACT_22]] : ui32
// CHECK:             %[[BITEXTRACT_23:.*]] = coredsl.bitextract %[[CONCAT_24]][447:416] : (ui480) -> ui32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_vec_y{{\[}}%[[CAST_32]] : ui4] = %[[BITEXTRACT_23]] : ui32
// CHECK:             %[[BITEXTRACT_24:.*]] = coredsl.bitextract %[[CONCAT_24]][479:448] : (ui480) -> ui32
// CHECK:             %[[CAST_34:.*]] = coredsl.cast %[[BITEXTRACT_24]] : ui32 to si32
// CHECK:             coredsl.set @OTHER_STRUCT_REGS_aStruct_notNested{{\[}}%[[CAST_32]] : ui4] = %[[CAST_34]] : si32
// CHECK:             %[[CAST_35:.*]] = hwarith.cast %[[CONSTANT_9]] : (si4) -> ui4
// CHECK:             %[[GET_51:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_35]] : ui4] : si32
// CHECK:             %[[GET_52:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_35]] : ui4] : ui32
// CHECK:             %[[GET_53:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_35]] : ui4] : ui32
// CHECK:             %[[CAST_36:.*]] = hwarith.cast %[[CONSTANT_8]] : (si4) -> ui4
// CHECK:             %[[GET_54:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_36]] : ui4] : si32
// CHECK:             %[[GET_55:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_36]] : ui4] : ui32
// CHECK:             %[[GET_56:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_36]] : ui4] : ui32
// CHECK:             %[[CAST_37:.*]] = hwarith.cast %[[CONSTANT_7]] : (si4) -> ui4
// CHECK:             %[[GET_57:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_37]] : ui4] : si32
// CHECK:             %[[GET_58:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_37]] : ui4] : ui32
// CHECK:             %[[GET_59:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_37]] : ui4] : ui32
// CHECK:             %[[CONCAT_25:.*]] = coredsl.concat %[[GET_51]], %[[GET_52]] : si32, ui32
// CHECK:             %[[CONCAT_26:.*]] = coredsl.concat %[[CONCAT_25]], %[[GET_53]] : ui64, ui32
// CHECK:             %[[CONCAT_27:.*]] = coredsl.concat %[[CONCAT_26]], %[[GET_54]] : ui96, si32
// CHECK:             %[[CONCAT_28:.*]] = coredsl.concat %[[CONCAT_27]], %[[GET_55]] : ui128, ui32
// CHECK:             %[[CONCAT_29:.*]] = coredsl.concat %[[CONCAT_28]], %[[GET_56]] : ui160, ui32
// CHECK:             %[[CONCAT_30:.*]] = coredsl.concat %[[CONCAT_29]], %[[GET_57]] : ui192, si32
// CHECK:             %[[CONCAT_31:.*]] = coredsl.concat %[[CONCAT_30]], %[[GET_58]] : ui224, ui32
// CHECK:             %[[CONCAT_32:.*]] = coredsl.concat %[[CONCAT_31]], %[[GET_59]] : ui256, ui32
// CHECK:             %[[CAST_38:.*]] = hwarith.cast %[[CONSTANT_8]] : (si4) -> ui4
// CHECK:             %[[CAST_39:.*]] = coredsl.cast %[[CONCAT_32]] : ui288 to ui32
// CHECK:             %[[CAST_40:.*]] = coredsl.cast %[[CAST_39]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_38]] : ui4] = %[[CAST_40]] : si32
// CHECK:             %[[BITEXTRACT_25:.*]] = coredsl.bitextract %[[CONCAT_32]][63:32] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_38]] : ui4] = %[[BITEXTRACT_25]] : ui32
// CHECK:             %[[BITEXTRACT_26:.*]] = coredsl.bitextract %[[CONCAT_32]][95:64] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_38]] : ui4] = %[[BITEXTRACT_26]] : ui32
// CHECK:             %[[CAST_41:.*]] = hwarith.cast %[[CONSTANT_7]] : (si4) -> ui4
// CHECK:             %[[BITEXTRACT_27:.*]] = coredsl.bitextract %[[CONCAT_32]][127:96] : (ui288) -> ui32
// CHECK:             %[[CAST_42:.*]] = coredsl.cast %[[BITEXTRACT_27]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_41]] : ui4] = %[[CAST_42]] : si32
// CHECK:             %[[BITEXTRACT_28:.*]] = coredsl.bitextract %[[CONCAT_32]][159:128] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_41]] : ui4] = %[[BITEXTRACT_28]] : ui32
// CHECK:             %[[BITEXTRACT_29:.*]] = coredsl.bitextract %[[CONCAT_32]][191:160] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_41]] : ui4] = %[[BITEXTRACT_29]] : ui32
// CHECK:             %[[CAST_43:.*]] = hwarith.cast %[[CONSTANT_6]] : (si5) -> ui5
// CHECK:             %[[BITEXTRACT_30:.*]] = coredsl.bitextract %[[CONCAT_32]][223:192] : (ui288) -> ui32
// CHECK:             %[[CAST_44:.*]] = coredsl.cast %[[BITEXTRACT_30]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_43]] : ui5] = %[[CAST_44]] : si32
// CHECK:             %[[BITEXTRACT_31:.*]] = coredsl.bitextract %[[CONCAT_32]][255:224] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_43]] : ui5] = %[[BITEXTRACT_31]] : ui32
// CHECK:             %[[BITEXTRACT_32:.*]] = coredsl.bitextract %[[CONCAT_32]][287:256] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_43]] : ui5] = %[[BITEXTRACT_32]] : ui32
// CHECK:             %[[GET_60:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CONSTANT_14]] : ui4] : si32
// CHECK:             %[[GET_61:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CONSTANT_14]] : ui4] : ui32
// CHECK:             %[[GET_62:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CONSTANT_14]] : ui4] : ui32
// CHECK:             %[[ADD_15:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_13]] : (ui4, si2) -> si6
// CHECK:             %[[CAST_45:.*]] = hwarith.cast %[[ADD_15]] : (si6) -> ui5
// CHECK:             %[[GET_63:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_45]] : ui5] : si32
// CHECK:             %[[GET_64:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_45]] : ui5] : ui32
// CHECK:             %[[GET_65:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_45]] : ui5] : ui32
// CHECK:             %[[CONCAT_33:.*]] = coredsl.concat %[[GET_60]], %[[GET_61]] : si32, ui32
// CHECK:             %[[CONCAT_34:.*]] = coredsl.concat %[[CONCAT_33]], %[[GET_62]] : ui64, ui32
// CHECK:             %[[CONCAT_35:.*]] = coredsl.concat %[[CONCAT_34]], %[[GET_63]] : ui96, si32
// CHECK:             %[[CONCAT_36:.*]] = coredsl.concat %[[CONCAT_35]], %[[GET_64]] : ui128, ui32
// CHECK:             %[[CONCAT_37:.*]] = coredsl.concat %[[CONCAT_36]], %[[GET_65]] : ui160, ui32
// CHECK:             %[[CAST_46:.*]] = coredsl.cast %[[CONCAT_37]] : ui192 to ui32
// CHECK:             %[[CAST_47:.*]] = coredsl.cast %[[CAST_46]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CONSTANT_14]] : ui4] = %[[CAST_47]] : si32
// CHECK:             %[[BITEXTRACT_33:.*]] = coredsl.bitextract %[[CONCAT_37]][63:32] : (ui192) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CONSTANT_14]] : ui4] = %[[BITEXTRACT_33]] : ui32
// CHECK:             %[[BITEXTRACT_34:.*]] = coredsl.bitextract %[[CONCAT_37]][95:64] : (ui192) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CONSTANT_14]] : ui4] = %[[BITEXTRACT_34]] : ui32
// CHECK:             %[[ADD_16:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_13]] : (ui4, si2) -> si6
// CHECK:             %[[CAST_48:.*]] = hwarith.cast %[[ADD_16]] : (si6) -> ui5
// CHECK:             %[[BITEXTRACT_35:.*]] = coredsl.bitextract %[[CONCAT_37]][127:96] : (ui192) -> ui32
// CHECK:             %[[CAST_49:.*]] = coredsl.cast %[[BITEXTRACT_35]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_48]] : ui5] = %[[CAST_49]] : si32
// CHECK:             %[[BITEXTRACT_36:.*]] = coredsl.bitextract %[[CONCAT_37]][159:128] : (ui192) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_48]] : ui5] = %[[BITEXTRACT_36]] : ui32
// CHECK:             %[[BITEXTRACT_37:.*]] = coredsl.bitextract %[[CONCAT_37]][191:160] : (ui192) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_48]] : ui5] = %[[BITEXTRACT_37]] : ui32
// CHECK:             %[[ADD_17:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_11]] : (ui4, si3) -> si6
// CHECK:             %[[CAST_50:.*]] = hwarith.cast %[[ADD_17]] : (si6) -> ui5
// CHECK:             %[[GET_66:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_50]] : ui5] : si32
// CHECK:             %[[GET_67:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_50]] : ui5] : ui32
// CHECK:             %[[GET_68:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_50]] : ui5] : ui32
// CHECK:             %[[ADD_18:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_10]] : (ui4, si4) -> si6
// CHECK:             %[[CAST_51:.*]] = hwarith.cast %[[ADD_18]] : (si6) -> ui5
// CHECK:             %[[GET_69:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_51]] : ui5] : si32
// CHECK:             %[[GET_70:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_51]] : ui5] : ui32
// CHECK:             %[[GET_71:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_51]] : ui5] : ui32
// CHECK:             %[[ADD_19:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_9]] : (ui4, si4) -> si6
// CHECK:             %[[CAST_52:.*]] = hwarith.cast %[[ADD_19]] : (si6) -> ui5
// CHECK:             %[[GET_72:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_52]] : ui5] : si32
// CHECK:             %[[GET_73:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_52]] : ui5] : ui32
// CHECK:             %[[GET_74:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_52]] : ui5] : ui32
// CHECK:             %[[CONCAT_38:.*]] = coredsl.concat %[[GET_66]], %[[GET_67]] : si32, ui32
// CHECK:             %[[CONCAT_39:.*]] = coredsl.concat %[[CONCAT_38]], %[[GET_68]] : ui64, ui32
// CHECK:             %[[CONCAT_40:.*]] = coredsl.concat %[[CONCAT_39]], %[[GET_69]] : ui96, si32
// CHECK:             %[[CONCAT_41:.*]] = coredsl.concat %[[CONCAT_40]], %[[GET_70]] : ui128, ui32
// CHECK:             %[[CONCAT_42:.*]] = coredsl.concat %[[CONCAT_41]], %[[GET_71]] : ui160, ui32
// CHECK:             %[[CONCAT_43:.*]] = coredsl.concat %[[CONCAT_42]], %[[GET_72]] : ui192, si32
// CHECK:             %[[CONCAT_44:.*]] = coredsl.concat %[[CONCAT_43]], %[[GET_73]] : ui224, ui32
// CHECK:             %[[CONCAT_45:.*]] = coredsl.concat %[[CONCAT_44]], %[[GET_74]] : ui256, ui32
// CHECK:             %[[CAST_53:.*]] = coredsl.cast %[[CONCAT_45]] : ui288 to ui96
// CHECK:             %[[BITEXTRACT_38:.*]] = coredsl.bitextract %[[CONCAT_45]][191:96] : (ui288) -> ui96
// CHECK:             %[[BITEXTRACT_39:.*]] = coredsl.bitextract %[[CONCAT_45]][287:192] : (ui288) -> ui96
// CHECK:             %[[CONCAT_46:.*]] = coredsl.concat %[[BITEXTRACT_39]], %[[BITEXTRACT_38]] : ui96, ui96
// CHECK:             %[[CONCAT_47:.*]] = coredsl.concat %[[CONCAT_46]], %[[CAST_53]] : ui192, ui96
// CHECK:             %[[CAST_54:.*]] = coredsl.cast %[[CONCAT_47]] : ui288 to ui96
// CHECK:             %[[BITEXTRACT_40:.*]] = coredsl.bitextract %[[CONCAT_47]][191:96] : (ui288) -> ui96
// CHECK:             %[[BITEXTRACT_41:.*]] = coredsl.bitextract %[[CONCAT_47]][287:192] : (ui288) -> ui96
// CHECK:             %[[CONCAT_48:.*]] = coredsl.concat %[[BITEXTRACT_41]], %[[BITEXTRACT_40]] : ui96, ui96
// CHECK:             %[[CONCAT_49:.*]] = coredsl.concat %[[CONCAT_48]], %[[CAST_54]] : ui192, ui96
// CHECK:             %[[ADD_20:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_11]] : (ui4, si3) -> si6
// CHECK:             %[[CAST_55:.*]] = hwarith.cast %[[ADD_20]] : (si6) -> ui5
// CHECK:             %[[CAST_56:.*]] = coredsl.cast %[[CONCAT_49]] : ui288 to ui32
// CHECK:             %[[CAST_57:.*]] = coredsl.cast %[[CAST_56]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_55]] : ui5] = %[[CAST_57]] : si32
// CHECK:             %[[BITEXTRACT_42:.*]] = coredsl.bitextract %[[CONCAT_49]][63:32] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_55]] : ui5] = %[[BITEXTRACT_42]] : ui32
// CHECK:             %[[BITEXTRACT_43:.*]] = coredsl.bitextract %[[CONCAT_49]][95:64] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_55]] : ui5] = %[[BITEXTRACT_43]] : ui32
// CHECK:             %[[ADD_21:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_10]] : (ui4, si4) -> si6
// CHECK:             %[[CAST_58:.*]] = hwarith.cast %[[ADD_21]] : (si6) -> ui5
// CHECK:             %[[BITEXTRACT_44:.*]] = coredsl.bitextract %[[CONCAT_49]][127:96] : (ui288) -> ui32
// CHECK:             %[[CAST_59:.*]] = coredsl.cast %[[BITEXTRACT_44]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_58]] : ui5] = %[[CAST_59]] : si32
// CHECK:             %[[BITEXTRACT_45:.*]] = coredsl.bitextract %[[CONCAT_49]][159:128] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_58]] : ui5] = %[[BITEXTRACT_45]] : ui32
// CHECK:             %[[BITEXTRACT_46:.*]] = coredsl.bitextract %[[CONCAT_49]][191:160] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_58]] : ui5] = %[[BITEXTRACT_46]] : ui32
// CHECK:             %[[ADD_22:.*]] = hwarith.add %[[CONSTANT_14]], %[[CONSTANT_9]] : (ui4, si4) -> si6
// CHECK:             %[[CAST_60:.*]] = hwarith.cast %[[ADD_22]] : (si6) -> ui5
// CHECK:             %[[BITEXTRACT_47:.*]] = coredsl.bitextract %[[CONCAT_49]][223:192] : (ui288) -> ui32
// CHECK:             %[[CAST_61:.*]] = coredsl.cast %[[BITEXTRACT_47]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_60]] : ui5] = %[[CAST_61]] : si32
// CHECK:             %[[BITEXTRACT_48:.*]] = coredsl.bitextract %[[CONCAT_49]][255:224] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_60]] : ui5] = %[[BITEXTRACT_48]] : ui32
// CHECK:             %[[BITEXTRACT_49:.*]] = coredsl.bitextract %[[CONCAT_49]][287:256] : (ui288) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_60]] : ui5] = %[[BITEXTRACT_49]] : ui32
// CHECK:             %[[CAST_62:.*]] = hwarith.cast %[[CONSTANT_5]] : (si1) -> ui1
// CHECK:             %[[GET_75:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_62]] : ui1] : si32
// CHECK:             %[[GET_76:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_62]] : ui1] : ui32
// CHECK:             %[[GET_77:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_62]] : ui1] : ui32
// CHECK:             %[[CAST_63:.*]] = hwarith.cast %[[CONSTANT_13]] : (si2) -> ui2
// CHECK:             %[[GET_78:.*]] = coredsl.get @STRUCT_REGS_notNested{{\[}}%[[CAST_63]] : ui2] : si32
// CHECK:             %[[GET_79:.*]] = coredsl.get @STRUCT_REGS_vec_x{{\[}}%[[CAST_63]] : ui2] : ui32
// CHECK:             %[[GET_80:.*]] = coredsl.get @STRUCT_REGS_vec_y{{\[}}%[[CAST_63]] : ui2] : ui32
// CHECK:             %[[CONCAT_50:.*]] = coredsl.concat %[[GET_75]], %[[GET_76]] : si32, ui32
// CHECK:             %[[CONCAT_51:.*]] = coredsl.concat %[[CONCAT_50]], %[[GET_77]] : ui64, ui32
// CHECK:             %[[CONCAT_52:.*]] = coredsl.concat %[[CONCAT_51]], %[[GET_78]] : ui96, si32
// CHECK:             %[[CONCAT_53:.*]] = coredsl.concat %[[CONCAT_52]], %[[GET_79]] : ui128, ui32
// CHECK:             %[[CONCAT_54:.*]] = coredsl.concat %[[CONCAT_53]], %[[GET_80]] : ui160, ui32
// CHECK:             %[[CAST_64:.*]] = hwarith.cast %[[CONSTANT_5]] : (si1) -> ui1
// CHECK:             %[[CAST_65:.*]] = coredsl.cast %[[CONCAT_54]] : ui192 to ui32
// CHECK:             %[[CAST_66:.*]] = coredsl.cast %[[CAST_65]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_64]] : ui1] = %[[CAST_66]] : si32
// CHECK:             %[[BITEXTRACT_50:.*]] = coredsl.bitextract %[[CONCAT_54]][63:32] : (ui192) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_64]] : ui1] = %[[BITEXTRACT_50]] : ui32
// CHECK:             %[[BITEXTRACT_51:.*]] = coredsl.bitextract %[[CONCAT_54]][95:64] : (ui192) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_64]] : ui1] = %[[BITEXTRACT_51]] : ui32
// CHECK:             %[[CAST_67:.*]] = hwarith.cast %[[CONSTANT_13]] : (si2) -> ui2
// CHECK:             %[[BITEXTRACT_52:.*]] = coredsl.bitextract %[[CONCAT_54]][127:96] : (ui192) -> ui32
// CHECK:             %[[CAST_68:.*]] = coredsl.cast %[[BITEXTRACT_52]] : ui32 to si32
// CHECK:             coredsl.set @STRUCT_REGS_notNested{{\[}}%[[CAST_67]] : ui2] = %[[CAST_68]] : si32
// CHECK:             %[[BITEXTRACT_53:.*]] = coredsl.bitextract %[[CONCAT_54]][159:128] : (ui192) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_x{{\[}}%[[CAST_67]] : ui2] = %[[BITEXTRACT_53]] : ui32
// CHECK:             %[[BITEXTRACT_54:.*]] = coredsl.bitextract %[[CONCAT_54]][191:160] : (ui192) -> ui32
// CHECK:             coredsl.set @STRUCT_REGS_vec_y{{\[}}%[[CAST_67]] : ui2] = %[[BITEXTRACT_54]] : ui32
// CHECK:             coredsl.end
// CHECK:           }
// CHECK:           coredsl.instruction @MultipleReturnValues("0000000", %[[VAL_10:.*]] : ui5, "00000000000000000000"){
// CHECK:             %[[CONSTANT_15:.*]] = hwarith.constant 1 : ui1
// CHECK:             %[[CAST_69:.*]] = coredsl.cast %[[CONSTANT_15]] : ui1 to i1
// CHECK:             %[[GET_81:.*]] = coredsl.get @STRUCT_REG_x : ui32
// CHECK:             %[[GET_82:.*]] = coredsl.get @STRUCT_REG_y : ui32
// CHECK:             %[[STRUCT_CREATE_0:.*]] = hw.struct_create (%[[GET_81]], %[[GET_82]]) : !hw.struct<x: ui32, y: ui32>
// CHECK:             %[[GET_83:.*]] = coredsl.get @OTHER_STRUCT_REG_x : ui32
// CHECK:             %[[GET_84:.*]] = coredsl.get @OTHER_STRUCT_REG_y : ui32
// CHECK:             %[[STRUCT_CREATE_1:.*]] = hw.struct_create (%[[GET_83]], %[[GET_84]]) : !hw.struct<x: ui32, y: ui32>
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[CAST_69]], %[[STRUCT_CREATE_1]], %[[STRUCT_CREATE_0]] : !hw.struct<x: ui32, y: ui32>
// CHECK:             %[[STRUCT_EXTRACT_2:.*]] = hw.struct_extract %[[SELECT_0]]["x"] : !hw.struct<x: ui32, y: ui32>
// CHECK:             coredsl.set @STRUCT_REG_x = %[[STRUCT_EXTRACT_2]] : ui32
// CHECK:             %[[STRUCT_EXTRACT_3:.*]] = hw.struct_extract %[[SELECT_0]]["y"] : !hw.struct<x: ui32, y: ui32>
// CHECK:             coredsl.set @STRUCT_REG_y = %[[STRUCT_EXTRACT_3]] : ui32
// CHECK:             coredsl.end
// CHECK:           }
// CHECK:         }
