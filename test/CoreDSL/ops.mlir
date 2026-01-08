// RUN: shortnail-opt %s -split-input-file -verify-diagnostics | shortnail-opt -verify-diagnostics | FileCheck %s

coredsl.isax "" {
  // CHECK-LABEL:   coredsl.instruction @bitwise_ops_tests(
  // CHECK-SAME:             "00", "000000000000000", "001", "00000", "0010111")
// CHECK:         {
// CHECK:           %[[_VAR_0:.*]] = hwarith.constant 1 : si2
// CHECK:           %[[_VAR_1:.*]] = hwarith.constant 7 : ui5
// CHECK:           %[[_VAR_2:.*]] = hwarith.constant 255 : ui8
// CHECK:           %[[_VAR_3:.*]] = hwarith.constant -1 : si1
// CHECK:           %[[_VAR_4:.*]] = coredsl.and %[[_VAR_2]], %[[_VAR_1]] : ui8, ui5
// CHECK:           %[[_VAR_5:.*]] = coredsl.or %[[_VAR_3]], %[[_VAR_0]] : si1, si2
// CHECK:           %[[_VAR_6:.*]] = coredsl.and %[[_VAR_1]], %[[_VAR_1]] : ui5, ui5
// CHECK:           %[[_VAR_7:.*]] = coredsl.xor %[[_VAR_2]], %[[_VAR_3]] : ui8, si1
// CHECK:           %[[_VAR_8:.*]] = hwarith.add %[[_VAR_4]], %[[_VAR_5]] : (ui8, si2) -> si10
// CHECK:           %[[_VAR_9:.*]] = hwarith.add %[[_VAR_6]], %[[_VAR_7]] : (ui5, si8) -> si9
// CHECK:           coredsl.end
// CHECK:         }
  coredsl.instruction @bitwise_ops_tests("00", "000000000000000", "001", "00000", "0010111") {
    %0 = hwarith.constant 1 : si2
    %1 = hwarith.constant 7 : ui5
    %2 = hwarith.constant 255 : ui8
    %3 = hwarith.constant -1 : si1

    %4 = coredsl.and %2, %1 : ui8, ui5
    %5 = coredsl.or %3, %0 : si1, si2
    // NOP
    %6 = coredsl.and %1, %1 : ui5, ui5
    %7 = coredsl.xor %2, %3 : ui8, si1

    // check the previous result types
    %8 = hwarith.add %4, %5 : (ui8, si2) -> si10
    %9 = hwarith.add %6, %7 : (ui5, si8) -> si9
    coredsl.end
  }
}

// -----

coredsl.isax "" {
  // CHECK-LABEL:   coredsl.instruction @shift_tests(
  // CHECK-SAME:             "00", "000000000000000", "001", "00000", "0010111")
// CHECK:         {
// CHECK:           %[[_VAR_0:.*]] = hwarith.constant 1 : si2
// CHECK:           %[[_VAR_1:.*]] = hwarith.constant 7 : ui5
// CHECK:           %[[_VAR_2:.*]] = hwarith.constant 255 : ui8
// CHECK:           %[[_VAR_3:.*]] = hwarith.constant -1 : si5
// CHECK:           %[[_VAR_4:.*]] = coredsl.shift_left %[[_VAR_2]], %[[_VAR_3]] : ui8, si5
// CHECK:           %[[_VAR_5:.*]] = coredsl.shift_right %[[_VAR_3]], %[[_VAR_0]] : si5, si2
// CHECK:           %[[_VAR_6:.*]] = coredsl.shift_right %[[_VAR_1]], %[[_VAR_2]] : ui5, ui8
// CHECK:           %[[_VAR_7:.*]] = coredsl.shift_right %[[_VAR_2]], %[[_VAR_3]] : ui8, si5
// CHECK:           %[[_VAR_8:.*]] = hwarith.add %[[_VAR_4]], %[[_VAR_5]] : (ui8, si5) -> si10
// CHECK:           %[[_VAR_9:.*]] = hwarith.add %[[_VAR_6]], %[[_VAR_7]] : (ui5, ui8) -> ui9
// CHECK:           coredsl.end
// CHECK:         }
  coredsl.instruction @shift_tests("00", "000000000000000", "001", "00000", "0010111") {
    %0 = hwarith.constant 1 : si2
    %1 = hwarith.constant 7 : ui5
    %2 = hwarith.constant 255 : ui8
    %3 = hwarith.constant -1 : si5

    // Note: negative shifts flip the direction
    %4 = coredsl.shift_left %2, %3 : ui8, si5
    %5 = coredsl.shift_right %3, %0 : si5, si2
    %6 = coredsl.shift_right %1, %2 : ui5, ui8
    // Note: negative shifts flip the direction
    %7 = coredsl.shift_right %2, %3 : ui8, si5

    // check the previous result types
    %8 = hwarith.add %4, %5 : (ui8, si5) -> si10
    %9 = hwarith.add %6, %7 : (ui5, ui8) -> ui9
    coredsl.end
  }
}

// -----

coredsl.isax "" {
  // CHECK:         coredsl.register local @ACCUM[4] : ui48
  coredsl.register local @ACCUM[4] : ui48

  // CHECK-LABEL:   coredsl.instruction @acc(
  // CHECK-SAME:                             %[[VAL_1:.*]] : ui2, "000000000000000", "000", %[[VAL_2:.*]] : ui5, %[[VAL_0:.*]] : si7)
  // CHECK:         {
  // CHECK:           %[[RES_0:.*]] = coredsl.get @ACCUM[%[[VAL_1]] : ui2] : ui48
  // CHECK:           %[[RES_1:.*]] = hwarith.add %[[VAL_2]], %[[VAL_0]] : (ui5, si7) -> si8
  // CHECK:           coredsl.set @ACCUM[%[[VAL_1]] : ui2] = %[[RES_0]] : ui48
  // CHECK:           coredsl.end
  // CHECK:         }
  coredsl.instruction @acc(%acc: ui2, "000000000000000", "000", %imm1: ui5, %imm2: si7) {
    %a = coredsl.get @ACCUM[%acc : ui2] : ui48
    %0 = hwarith.add %imm1, %imm2 : (ui5, si7) -> si8

    coredsl.set @ACCUM[%acc : ui2] = %a : ui48
    coredsl.end
  }
}

// -----

coredsl.isax "" {
  // CHECK-LABEL:   coredsl.instruction @bitset_bitextract_tests(
  // CHECK-SAME:              "00", "000000000000000", "001", "00000", "0010111")
  // CHECK:         {
  // CHECK:           %[[RES_0:.*]] = hwarith.constant 0 : ui48
  // CHECK:           %[[RES_1:.*]] = hwarith.constant 0 : si48
  // CHECK:           %[[RES_2:.*]] = coredsl.bitextract %[[RES_0]][1] : (ui48) -> ui1
  // CHECK:           %[[RES_3:.*]] = coredsl.bitextract %[[RES_1]][0:15] : (si48) -> ui16
  // CHECK:           %[[RES_4:.*]] = coredsl.bitset %[[RES_0]][47:32] = %[[RES_3]] : (ui48, ui16) -> ui48
  // CHECK:           %[[RES_5:.*]] = coredsl.bitset %[[RES_1]][2:2] = %[[RES_2]] : (si48, ui1) -> si48
  // CHECK:           %[[RES_6:.*]] = hwarith.add %[[RES_2]], %[[RES_4]] : (ui1, ui48) -> ui49
  // CHECK:           %[[RES_7:.*]] = hwarith.add %[[RES_3]], %[[RES_5]] : (ui16, si48) -> si49
  // CHECK:           coredsl.end
  // CHECK:         }
  coredsl.instruction @bitset_bitextract_tests("00", "000000000000000", "001", "00000", "0010111") {
    %dummy = hwarith.constant 0 : ui48
    %signed_dummy = hwarith.constant 0 : si48

    %bit1 = coredsl.bitextract %dummy[1] : (ui48) -> ui1
    %multi_bit = coredsl.bitextract %signed_dummy[0:15] : (si48) -> ui16
    %res1 = coredsl.bitset %dummy[47:32] = %multi_bit : (ui48, ui16) -> ui48
    %res2 = coredsl.bitset %signed_dummy[2:2] = %bit1 : (si48, ui1) -> si48

    // check the previous result types
    %0 = hwarith.add %bit1, %res1 : (ui1, ui48) -> ui49
    %1 = hwarith.add %multi_bit, %res2 : (ui16, si48) -> si49
    coredsl.end
  }
}

// -----

coredsl.isax "" {
  // CHECK-LABEL:   coredsl.instruction @cast_tests(
  // CHECK-SAME:              "00", "000000000000000", "001", "00000", "0010111")
// CHECK:         {
// CHECK:           %[[_VAR_0:.*]] = hwarith.constant 255 : ui8
// CHECK:           %[[_VAR_1:.*]] = coredsl.cast %[[_VAR_0]] : ui8 to si3
// CHECK:           %[[_VAR_2:.*]] = coredsl.cast %[[_VAR_0]] : ui8 to ui8
// CHECK:           %[[_VAR_3:.*]] = coredsl.cast %[[_VAR_0]] : ui8 to si32
// CHECK:           %[[_VAR_4:.*]] = coredsl.cast %[[_VAR_0]] : ui8 to ui32
// CHECK:           %[[_VAR_5:.*]] = hwarith.add %[[_VAR_1]], %[[_VAR_2]] : (si3, ui8) -> si10
// CHECK:           %[[_VAR_6:.*]] = hwarith.add %[[_VAR_3]], %[[_VAR_4]] : (si32, ui32) -> si34
// CHECK:           coredsl.end
// CHECK:         }
  coredsl.instruction @cast_tests("00", "000000000000000", "001", "00000", "0010111") {
    %0 = hwarith.constant 255 : ui8

    %1 = coredsl.cast %0 : ui8 to si3
    // NOP
    %2 = coredsl.cast %0 : ui8 to ui8
    %3 = coredsl.cast %0 : ui8 to si32
    %4 = coredsl.cast %0 : ui8 to ui32

    // check the previous result types
    %5 = hwarith.add %1, %2 : (si3, ui8) -> si10
    %6 = hwarith.add %3, %4 : (si32, ui32) -> si34
    coredsl.end
  }
}

// -----

coredsl.isax "" {
  // CHECK-LABEL:   coredsl.instruction @concat_tests(
  // CHECK-SAME:              "00", "000000000000000", "001", "00000", "0010111")
// CHECK:         {
// CHECK:           %[[_VAR_0:.*]] = hwarith.constant 255 : ui8
// CHECK:           %[[_VAR_1:.*]] = hwarith.constant -1 : si3
// CHECK:           %[[_VAR_2:.*]] = coredsl.concat %[[_VAR_0]], %[[_VAR_1]] : ui8, si3
// CHECK:           %[[_VAR_3:.*]] = coredsl.concat %[[_VAR_1]], %[[_VAR_1]] : si3, si3
// CHECK:           %[[_VAR_4:.*]] = hwarith.add %[[_VAR_2]], %[[_VAR_3]] : (ui11, ui6) -> ui12
// CHECK:           coredsl.end
// CHECK:         }
  coredsl.instruction @concat_tests("00", "000000000000000", "001", "00000", "0010111") {
    %0 = hwarith.constant 255 : ui8
    %1 = hwarith.constant -1 : si3

    %2 = coredsl.concat %0, %1 : ui8, si3
    %3 = coredsl.concat %1, %1 : si3, si3

    // check the previous result types
    %4 = hwarith.add %2, %3 : (ui11, ui6) -> ui12
    coredsl.end
  }
}

// -----

coredsl.isax "" {
// CHECK-LABEL:   coredsl.instruction @mod_tests(
// CHECK-SAME:             "00", "000000000000000", "001", "00000", "0010111")
// CHECK:         {
// CHECK:           %[[VAR_0:.*]] = hwarith.constant 2 : ui2
// CHECK:           %[[VAR_1:.*]] = hwarith.constant 2 : si5
// CHECK:           %[[VAR_2:.*]] = hwarith.constant 255 : ui8
// CHECK:           %[[VAR_3:.*]] = hwarith.constant -1 : si1
// CHECK:           %[[VAR_4:.*]] = coredsl.mod %[[VAR_0]], %[[VAR_2]] : ui2, ui8
// CHECK:           %[[VAR_5:.*]] = coredsl.mod %[[VAR_1]], %[[VAR_0]] : si5, ui2
// CHECK:           %[[VAR_6:.*]] = coredsl.mod %[[VAR_2]], %[[VAR_3]] : ui8, si1
// CHECK:           %[[VAR_7:.*]] = hwarith.add %[[VAR_4]], %[[VAR_5]] : (ui2, si3) -> si4
// CHECK:           %[[VAR_8:.*]] = hwarith.add %[[VAR_6]], %[[VAR_7]] : (ui1, si4) -> si5
// CHECK:           coredsl.end

  coredsl.instruction @mod_tests("00", "000000000000000", "001", "00000", "0010111") {
    %0 = hwarith.constant 2 : ui2
    %1 = hwarith.constant 2 : si5
    %2 = hwarith.constant 255 : ui8
    %3 = hwarith.constant -1 : si1

    %4 = coredsl.mod %0, %2 : ui2, ui8
    // mixed signedness edge case 1
    %5 = coredsl.mod %1, %0 : si5, ui2
    // mixed signedness edge case 2
    %6 = coredsl.mod %2, %3 : ui8, si1

    // check the previous result types
    %7 = hwarith.add %4, %5 : (ui2, si3) -> si4
    %8 = hwarith.add %6, %7 : (ui1, si4) -> si5
    coredsl.end
  }
}
