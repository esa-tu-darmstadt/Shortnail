// RUN: shortnail-opt %s -coredsl-explode-struct-registers -canonicalize | shortnail-opt | FileCheck %s

// TODO: add test with triple nested struct
coredsl.isax "StructRegisters" {
  coredsl.register local @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
  coredsl.register local @NESTED_STRUCT_REG : !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>
  coredsl.register local @TRIPLE_NESTED_REG : !hw.struct<internalStruct: !hw.struct<notNested: si32, vec: !hw.struct<x: ui32, y: ui32>>, intVal: ui32>
// CHECK: coredsl.register local @STRUCT_REG_x  : ui32
// CHECK: coredsl.register local @STRUCT_REG_y  : ui32
// CHECK: coredsl.register local @NESTED_STRUCT_REG_notNested  : si32
// CHECK: coredsl.register local @NESTED_STRUCT_REG_vec_x  : ui32
// CHECK: coredsl.register local @NESTED_STRUCT_REG_vec_y  : ui32
// CHECK: coredsl.register local @TRIPLE_NESTED_REG_internalStruct_notNested  : si32
// CHECK: coredsl.register local @TRIPLE_NESTED_REG_internalStruct_vec_x  : ui32
// CHECK: coredsl.register local @TRIPLE_NESTED_REG_internalStruct_vec_y  : ui32
// CHECK: coredsl.register local @TRIPLE_NESTED_REG_intVal  : ui32

  coredsl.instruction @StructRegDirectStore {lil.enc_immediates = [[["%TREENAIL_WAS_HERE_imm_11_0", 11, 0, 0, "imm"]], [["%TREENAIL_WAS_HERE_rs1_4_0", 4, 0, 0, "rs1"]], [["%TREENAIL_WAS_HERE_rd_4_0", 4, 0, 0, "rd"]]]} (%TREENAIL_WAS_HERE_imm_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0000011") {
// CHECK: %0 = hwarith.constant 0 : ui1
// CHECK: %1 = hwarith.constant 255 : ui8
// CHECK: %2 = hwarith.constant 7 : ui3
// CHECK: %3 = hwarith.constant 1 : ui1
// CHECK: %4 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
// CHECK: %5 = coredsl.get @STRUCT_REG_x : ui32
// CHECK: %6 = coredsl.get @STRUCT_REG_y : ui32
// CHECK: %7 = coredsl.cast %4 : ui5 to ui32
// CHECK: coredsl.set @STRUCT_REG_x = %7 : ui32
// CHECK: coredsl.set @STRUCT_REG_y = %6 : ui32
// CHECK: %8 = coredsl.get @STRUCT_REG_x : ui32
// CHECK: %9 = coredsl.get @STRUCT_REG_y : ui32
// CHECK: %10 = hwarith.sub %8, %3 : (ui32, ui1) -> si33
// CHECK: %11 = coredsl.get @STRUCT_REG_x : ui32
// CHECK: %12 = coredsl.get @STRUCT_REG_y : ui32
// CHECK: %13 = coredsl.cast %10 : si33 to ui32
// CHECK: coredsl.set @STRUCT_REG_x = %13 : ui32
// CHECK: coredsl.set @STRUCT_REG_y = %12 : ui32
// CHECK: %14 = coredsl.get @NESTED_STRUCT_REG_notNested : si32
// CHECK: %15 = coredsl.get @NESTED_STRUCT_REG_vec_x : ui32
// CHECK: %16 = coredsl.get @NESTED_STRUCT_REG_vec_y : ui32
// CHECK: %17 = coredsl.cast %2 : ui3 to ui32
// CHECK: coredsl.set @NESTED_STRUCT_REG_notNested = %14 : si32
// CHECK: coredsl.set @NESTED_STRUCT_REG_vec_x = %17 : ui32
// CHECK: coredsl.set @NESTED_STRUCT_REG_vec_y = %16 : ui32
// CHECK: %18 = coredsl.get @STRUCT_REG_x : ui32
// CHECK: %19 = coredsl.get @STRUCT_REG_y : ui32
// CHECK: %20 = coredsl.bitset %18[7:0] = %1 : (ui32, ui8) -> ui32
// CHECK: coredsl.set @STRUCT_REG_x = %20 : ui32
// CHECK: coredsl.set @STRUCT_REG_y = %19 : ui32
// CHECK: %21 = coredsl.get @NESTED_STRUCT_REG_notNested : si32
// CHECK: %22 = coredsl.get @NESTED_STRUCT_REG_vec_x : ui32
// CHECK: %23 = coredsl.get @NESTED_STRUCT_REG_vec_y : ui32
// CHECK: %24 = coredsl.cast %0 : ui1 to ui4
// CHECK: %25 = coredsl.bitset %23[3:0] = %24 : (ui32, ui4) -> ui32
// CHECK: coredsl.set @NESTED_STRUCT_REG_notNested = %21 : si32
// CHECK: coredsl.set @NESTED_STRUCT_REG_vec_x = %22 : ui32
// CHECK: coredsl.set @NESTED_STRUCT_REG_vec_y = %25 : ui32
    %imm = coredsl.cast %TREENAIL_WAS_HERE_imm_11_0 : ui12 to ui12
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %0 = coredsl.get @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
    %1 = coredsl.cast %rs1 : ui5 to ui32
    %2 = hw.struct_inject %0["x"], %1 : !hw.struct<x: ui32, y: ui32>
    coredsl.set @STRUCT_REG = %2 : !hw.struct<x: ui32, y: ui32>
    %3 = hwarith.constant 1 : ui1
    %4 = coredsl.get @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
    %5 = hw.struct_extract %4["x"] : !hw.struct<x: ui32, y: ui32>
    %6 = hwarith.sub %5, %3 : (ui32, ui1) -> si33
    %7 = coredsl.get @STRUCT_REG : !hw.struct<x: ui32, y: ui32>
    %8 = coredsl.cast %6 : si33 to ui32
    %9 = hw.struct_inject %7["x"], %8 : !hw.struct<x: ui32, y: ui32>
    coredsl.set @STRUCT_REG = %9 : !hw.struct<x: ui32, y: ui32>
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
    coredsl.end
  }
}
