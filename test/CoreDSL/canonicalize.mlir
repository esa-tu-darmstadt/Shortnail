// RUN: shortnail-opt %s -canonicalize -split-input-file -verify-diagnostics | shortnail-opt | FileCheck %s

// CHECK:    coredsl.register local @ACC1 = 0 : ui48
// CHECK:    coredsl.register core_pc @ACC2 : ui48

coredsl.isax "" {
  coredsl.register local @ACC1[1] =0 : ui48
  coredsl.register core_pc @ACC2[1] : ui48
}

// -----

coredsl.isax "" {

// CHECK:  %[[RES_1:.*]] = hwarith.constant 0 : ui1
// CHECK:  %[[RES_0:.*]] = hwarith.constant 0 : si48
// CHECK:  coredsl.register local @TEST : si48
// CHECK:  coredsl.register local @TEST2 : ui32
// CHECK:  %[[RES_2:.*]] = coredsl.bitset %[[RES_0]][2] = %[[RES_1]] : (si48, ui1) -> si48
// CHECK:  %[[RES_3:.*]] = coredsl.cast %[[RES_0]] : si48 to ui32
// CHECK:  %[[RES_4:.*]] = coredsl.bitextract %[[RES_0]][0:31] : (si48) -> ui32
// CHECK:  coredsl.set @TEST = %[[RES_2]] : si48
// CHECK:  coredsl.set @TEST2 = %[[RES_3]] : ui32
// CHECK:  coredsl.set @TEST2 = %[[RES_4]] : ui32

  coredsl.register local @TEST : si48
  coredsl.register local @TEST2 : ui32
  %signed_dummy = hwarith.constant 0 : si48
  %bit = hwarith.constant 0 : ui1
  %res2 = coredsl.bitset %signed_dummy[2:2] = %bit : (si48, ui1) -> si48
  %res3 = coredsl.bitextract %signed_dummy[31:0]: (si48) -> ui32
  %res4 = coredsl.bitextract %signed_dummy[0:31]: (si48) -> ui32
  coredsl.set @TEST = %res2 : si48
  coredsl.set @TEST2 = %res3 : ui32
  coredsl.set @TEST2 = %res4 : ui32
}

// -----

// CHECK: %[[VAR_0:.*]] = hwarith.constant 0 : ui7
// CHECK: %[[VAR_1:.*]] = hwarith.constant 0 : ui3
// CHECK: coredsl.addrspace core_mem   @MEM : (ui32) -> ui8
// CHECK: %[[VAR_2:.*]] = coredsl.get @MEM[%[[VAR_1]] : ui3, 1] : ui8
// CHECK: %[[VAR_3:.*]] = coredsl.bitextract %[[VAR_2]][%[[VAR_1]] : ui3] : (ui8) -> ui1
// CHECK: %[[VAR_4:.*]] = coredsl.concat %[[VAR_3]], %[[VAR_0]] : ui1, ui7
// CHECK: %[[VAR_5:.*]] = coredsl.bitset %[[VAR_4]][%[[VAR_1]] : ui3, 1] = %[[VAR_3]] : (ui8, ui1) -> ui8
// CHECK: coredsl.set @MEM[%[[VAR_1]] : ui3] = %[[VAR_5]] : ui8

coredsl.isax "" {
  coredsl.addrspace core_mem @MEM : (ui32) -> ui8

  %idx = hwarith.constant 0 : ui3
  %stuff = hwarith.constant 0 : ui7
  %0 = coredsl.get @MEM[%idx: ui3, 1:1] : ui8

  %bit1 = coredsl.bitextract %0[%idx : ui3, 0:0] : (ui8) -> ui1
  %res = coredsl.concat %bit1, %stuff : ui1, ui7
  %bitset = coredsl.bitset %res[%idx : ui3, 1:1] = %bit1 : (ui8, ui1) -> ui8

  coredsl.set @MEM[%idx: ui3, 0:0] = %bitset : ui8
}
// CHECK-NEXT: }

// -----

// CHECK: coredsl.isax "" {
// CHECK:         %[[VAL_0:.*]] = hwarith.constant 4 : ui32
// CHECK:         %[[VAL_1:.*]] = hwarith.constant 3 : ui32
// CHECK:         %[[VAL_2:.*]] = hwarith.constant 2 : ui32
// CHECK:         %[[VAL_3:.*]] = hwarith.constant 1 : ui32
// CHECK:         %[[VAL_4:.*]] = hwarith.constant 43 : ui32
// CHECK:         %[[VAL_5:.*]] = hwarith.constant 42 : ui32
// CHECK:         %[[VAL_6:.*]] = hwarith.constant 0 : ui32
// CHECK:         coredsl.register local const @C1 = 0 : ui32
// CHECK:         coredsl.register local const @C2 = 42 : ui32
// CHECK:         coredsl.register local const @C3 = 43 : ui32
// CHECK:         coredsl.register local const @C4[4] = [1, 2, 3, 4] : ui32
// CHECK:         coredsl.alias @A1 = @C4[3:0]
// CHECK:         coredsl.alias @A2 = @C4[3:1]
// CHECK:         coredsl.alias @A3 = @C4[2:1]
// CHECK:         coredsl.alias @A4 = @C4[1:1]
// CHECK:         coredsl.register local @DUMMY_RESULT_SINK : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_6]] : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_5]] : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_4]] : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_3]] : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_2]] : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_1]] : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_0]] : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_2]] : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_1]] : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_2]] : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_1]] : ui32
// CHECK:         coredsl.set @DUMMY_RESULT_SINK = %[[VAL_2]] : ui32

coredsl.isax "" {
  coredsl.register local const @C1 = 0 : ui32
  coredsl.register local const @C2 = 42 : ui32
  coredsl.register local const @C3 = 43 : ui32
  coredsl.register local const @C4[4] = [1, 2, 3, 4] : ui32
  coredsl.alias @A1 = @C4[3:0]
  coredsl.alias @A2 = @C4[3:1]
  coredsl.alias @A3 = @C4[2:1]
  coredsl.alias @A4 = @C4[1:1]
  coredsl.register local @DUMMY_RESULT_SINK : ui32

  %idx = hwarith.constant 0 : ui2
  %idx1 = hwarith.constant 1 : ui2
  %0 = coredsl.get @C1 : ui32
  coredsl.set @DUMMY_RESULT_SINK = %0 : ui32
  %1 = coredsl.get @C2 : ui32
  coredsl.set @DUMMY_RESULT_SINK = %1 : ui32
  %2 = coredsl.get @C3 : ui32
  coredsl.set @DUMMY_RESULT_SINK = %2 : ui32
  %3 = coredsl.get @C4[0] : ui32
  coredsl.set @DUMMY_RESULT_SINK = %3 : ui32
  %4 = coredsl.get @C4[%idx : ui2, 1:1] : ui32
  coredsl.set @DUMMY_RESULT_SINK = %4 : ui32
  %5 = coredsl.get @C4[%idx1 : ui2, 1:1] : ui32
  coredsl.set @DUMMY_RESULT_SINK = %5 : ui32
  %6 = coredsl.get @A1[3] : ui32
  coredsl.set @DUMMY_RESULT_SINK = %6 : ui32
  %7 = coredsl.get @A1[%idx : ui2, 1:1] : ui32
  coredsl.set @DUMMY_RESULT_SINK = %7 : ui32
  %8 = coredsl.get @A1[%idx1 : ui2, 1:1] : ui32
  coredsl.set @DUMMY_RESULT_SINK = %8 : ui32
  %9 = coredsl.get @A2[0] : ui32
  coredsl.set @DUMMY_RESULT_SINK = %9 : ui32
  %10 = coredsl.get @A3[1] : ui32
  coredsl.set @DUMMY_RESULT_SINK = %10 : ui32
  %11 = coredsl.get @A4 : ui32
  coredsl.set @DUMMY_RESULT_SINK = %11 : ui32
}
