// RUN: shortnail-opt %s -split-input-file -verify-diagnostics | shortnail-opt | FileCheck %s

// CHECK:         coredsl.register core_pc @PC : ui32
// CHECK:         coredsl.register local @ACC : ui48
// CHECK:         coredsl.register local @ACC1[1] = 0 : ui48
// CHECK:         coredsl.register core_pc @ACC2[1] = 0 : ui48
// CHECK:         coredsl.register local @ACC3 = 1 : ui48
// CHECK:         coredsl.register core_pc @ACC4 = 2 : ui48
// CHECK:         coredsl.register local @ACC5 = 255 : ui8
// CHECK:         coredsl.register core_pc @ACC6 = -128 : si8

// CHECK-LABEL:   coredsl.instruction @reset_accum(
// CHECK-SAME:         %[[VAL_0:.*]] : ui2, "000000000000000", "001", "00000", "0010111")
// CHECK:          {
// CHECK:           %[[VAL_1:.*]] = hwarith.constant 0 : ui48
// CHECK:           %[[VAL_2:.*]] = coredsl.get @ACC : ui48
// CHECK:           coredsl.set @ACC = %[[VAL_1]] : ui48
// CHECK:           coredsl.end
// CHECK:         }

coredsl.isax "" {
    coredsl.register core_pc @PC : ui32
    coredsl.register local @ACC : ui48

    // Legal init value expressions:
    coredsl.register local @ACC1[1] =0 : ui48
    coredsl.register core_pc @ACC2[1]= 0 : ui48
    coredsl.register local @ACC3 = 1: ui48
    coredsl.register core_pc @ACC4 = 2 :ui48
    coredsl.register local @ACC5= 255 : ui8
    coredsl.register core_pc @ACC6=-128: si8

    coredsl.instruction @reset_accum(%acc : ui2, "000000000000000", "001", "00000", "0010111") {
      %zero = hwarith.constant 0 : ui48
      %dummy = coredsl.get @ACC : ui48
      coredsl.set @ACC = %zero : ui48
      coredsl.end
    }
}

// -----

// test different register keyword combinations
// CHECK:         coredsl.register core_pc @T2 : ui32
// CHECK:         coredsl.register core_pc @T6 = 12 : ui32
coredsl.isax "" {
    coredsl.register core_pc @T2 : ui32
    coredsl.register core_pc @T6 = 12 : ui32
}

// -----

coredsl.isax "" {
  // expected-error @+1 {{Const registers must be initialized}}
  coredsl.register local const @C1 : ui32
}
