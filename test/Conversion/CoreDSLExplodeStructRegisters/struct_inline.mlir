// This tests whether function inlining works in combination with structs,
// because the inliner emits hw.constant ops with non-signless integer types
// when folding hw.struct_extract, which is invalid and needed to be corrected
// by a cleanup canonicalizer pattern in shortnail

// RUN: shortnail-opt %s -inline -canonicalize -cse | FileCheck %s

// CHECK-LABEL:   coredsl.isax "FunctionsWithStructArgInline" {
// CHECK:           func.func @convertSimple(%[[VAL_0:.*]]: !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>) -> ui64 {
// CHECK:             %[[STRUCT_EXTRACT_0:.*]] = hw.struct_extract %[[VAL_0]]["x"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
// CHECK:             %[[STRUCT_EXTRACT_1:.*]] = hw.struct_extract %[[VAL_0]]["y"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
// CHECK:             %[[MUL_0:.*]] = hwarith.mul %[[STRUCT_EXTRACT_0]], %[[STRUCT_EXTRACT_1]] : (ui32, ui32) -> ui64
// CHECK:             %[[STRUCT_EXTRACT_2:.*]] = hw.struct_extract %[[VAL_0]]["a"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
// CHECK:             %[[STRUCT_EXTRACT_3:.*]] = hw.struct_extract %[[VAL_0]]["b"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
// CHECK:             %[[MUL_1:.*]] = hwarith.mul %[[STRUCT_EXTRACT_2]], %[[STRUCT_EXTRACT_3]] : (si16, si16) -> si32
// CHECK:             %[[CAST_0:.*]] = coredsl.cast %[[MUL_1]] : si32 to ui32
// CHECK:             %[[ADD_0:.*]] = hwarith.add %[[MUL_0]], %[[CAST_0]] : (ui64, ui32) -> ui65
// CHECK:             %[[CAST_1:.*]] = coredsl.cast %[[ADD_0]] : ui65 to ui64
// CHECK:             %[[STRUCT_EXTRACT_4:.*]] = hw.struct_extract %[[VAL_0]]["c"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
// CHECK:             %[[STRUCT_EXTRACT_5:.*]] = hw.struct_extract %[[VAL_0]]["d"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
// CHECK:             %[[DIV_0:.*]] = hwarith.div %[[STRUCT_EXTRACT_4]], %[[STRUCT_EXTRACT_5]] : (si16, si16) -> si17
// CHECK:             %[[CAST_2:.*]] = coredsl.cast %[[DIV_0]] : si17 to ui32
// CHECK:             %[[MUL_2:.*]] = hwarith.mul %[[CAST_1]], %[[CAST_2]] : (ui64, ui32) -> ui96
// CHECK:             %[[CAST_3:.*]] = coredsl.cast %[[MUL_2]] : ui96 to ui64
// CHECK:             return %[[CAST_3]] : ui64
// CHECK:           }
// CHECK:           coredsl.instruction @StructLocalVals {lil.enc_immediates = {{\[\[}}["%[[VAL_1:.*]]", 11, 0, 0, "imm"]], {{\[\[}}"%[[VAL_2:.*]]", 4, 0, 0, "rs1"]], {{\[\[}}"%[[VAL_3:.*]]", 4, 0, 0, "rd"]]]}(%[[VAL_1]] : ui12, %[[VAL_2]] : ui5, "010", %[[VAL_3]] : ui5, "0000011"){
// CHECK:             coredsl.end
// CHECK:           }
// CHECK:         }


coredsl.isax "FunctionsWithStructArgInline" {
  func.func @convertSimple(%arg : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>) -> ui64 {
    %0 = hw.struct_extract %arg["x"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
    %1 = hw.struct_extract %arg["y"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
    %2 = hwarith.mul %0, %1 : (ui32, ui32) -> ui64
    %3 = hw.struct_extract %arg["a"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
    %4 = hw.struct_extract %arg["b"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
    %5 = hwarith.mul %3, %4 : (si16, si16) -> si32
    %6 = coredsl.cast %5 : si32 to ui32
    %7 = hwarith.add %2, %6 : (ui64, ui32) -> ui65
    %8 = coredsl.cast %7 : ui65 to ui64
    %9 = hw.struct_extract %arg["c"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
    %10 = hw.struct_extract %arg["d"] : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
    %11 = hwarith.div %9, %10 : (si16, si16) -> si17
    %12 = coredsl.cast %11 : si17 to ui32
    %13 = hwarith.mul %8, %12 : (ui64, ui32) -> ui96
    %14 = coredsl.cast %13 : ui96 to ui64
    return %14 : ui64
  }
  coredsl.instruction @StructLocalVals {lil.enc_immediates = [[["%TREENAIL_WAS_HERE_imm_11_0", 11, 0, 0, "imm"]], [["%TREENAIL_WAS_HERE_rs1_4_0", 4, 0, 0, "rs1"]], [["%TREENAIL_WAS_HERE_rd_4_0", 4, 0, 0, "rd"]]]} (%TREENAIL_WAS_HERE_imm_11_0 : ui12, %TREENAIL_WAS_HERE_rs1_4_0 : ui5, "010", %TREENAIL_WAS_HERE_rd_4_0 : ui5, "0000011") {
    %imm = coredsl.cast %TREENAIL_WAS_HERE_imm_11_0 : ui12 to ui12
    %rs1 = coredsl.cast %TREENAIL_WAS_HERE_rs1_4_0 : ui5 to ui5
    %rd = coredsl.cast %TREENAIL_WAS_HERE_rd_4_0 : ui5 to ui5
    %70 = hwarith.constant 6 : ui3
    %71 = hwarith.constant 7 : ui3
    %72 = coredsl.cast %70 : ui3 to ui32
    %73 = coredsl.cast %71 : ui3 to ui32
    %0 = hwarith.constant 0 : ui32
    %1 = hwarith.constant 0 : ui32
    %2 = hwarith.constant 0 : si16
    %3 = hwarith.constant 0 : si16
    %4 = hwarith.constant 0 : si16
    %5 = hwarith.constant 0 : si16
    %6 = hw.struct_create (%0, %1, %2, %3, %4, %5) : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
    %7 = hw.struct_inject %6["x"], %72 : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
    %8 = hw.struct_inject %7["y"], %73 : !hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>
    %75 = func.call @convertSimple(%8) : (!hw.struct<x: ui32, y: ui32, a: si16, b: si16, c: si16, d: si16>) -> ui64
    coredsl.end
  }
}
