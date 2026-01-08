// RUN: shortnail-opt %s -canonicalize -split-input-file -verify-diagnostics | shortnail-opt | FileCheck %s

// CHECK:         coredsl.register core_x @X[32] : ui32
// CHECK:         coredsl.register local @ACCUM[5] : ui32
// CHECK:         coredsl.register local @PSEUDO_REG_FIELD : ui32

// CHECK-LABEL:   coredsl.instruction @nop(
// CHECK-SAME:                             "00", "000000000000000", "000", "00000", "0000000")
// CHECK:          {
// CHECK:           %[[VAL_4:.*]] = hwarith.constant 0 : ui3
// CHECK:           %[[VAL_3:.*]] = hwarith.constant 0 : ui2
// CHECK:           %[[VAL_1:.*]] = hwarith.constant 0 : ui32

// CHECK:           coredsl.set @ACCUM[%[[VAL_3]] : ui2] = %[[VAL_1]] : ui32
// CHECK:           %[[VAL_5:.*]] = coredsl.get @ACCUM[%[[VAL_3]] : ui2] : ui32

// CHECK:           coredsl.set @ACCUM[%[[VAL_4]] : ui3] = %[[VAL_1]] : ui32
// CHECK:           %[[VAL_6:.*]] = coredsl.get @ACCUM[%[[VAL_4]] : ui3] : ui32

// CHECK:           coredsl.set @PSEUDO_REG_FIELD = %[[VAL_1]] : ui32
// CHECK:           %[[VAL_7:.*]] = coredsl.get @PSEUDO_REG_FIELD : ui32

// CHECK:           coredsl.end
// CHECK:         }

coredsl.isax "" {
    coredsl.register core_x @X[32] : ui32
    coredsl.register local @ACCUM[5] : ui32
    coredsl.register local @PSEUDO_REG_FIELD[1] : ui32

    coredsl.instruction @nop("00", "000000000000000", "000", "00000", "0000000") {
      %idx_3 = hwarith.constant 0 : ui3
      %idx_2 = hwarith.constant 0 : ui2
      %zero = hwarith.constant 0 : ui32

      coredsl.set @ACCUM[%idx_2 : ui2] = %zero : ui32
      %dummy1 = coredsl.get @ACCUM[%idx_2 : ui2] : ui32

      coredsl.set @ACCUM[%idx_3 : ui3] = %zero : ui32
      %dummy2 = coredsl.get @ACCUM[%idx_3 : ui3] : ui32

      coredsl.set @PSEUDO_REG_FIELD = %zero : ui32
      %dummy3 = coredsl.get @PSEUDO_REG_FIELD : ui32

      coredsl.end
    }
}

// -----

// test different register keyword combinations
// CHECK:         coredsl.register core_x const volatile @T1[3] = [0, 1, 2] : ui32
// CHECK:         coredsl.register local volatile @T2[42] : ui32
coredsl.isax "" {
    // TODO that's stupid, why would we want this?
    coredsl.register core_x volatile const @T1[3] = [0, 1, 2] : ui32
    coredsl.register local volatile @T2[42] : ui32
}

// -----

// CHECK:         coredsl.register core_x @ACCUM[4] : ui48

// CHECK-LABEL:   coredsl.instruction @reset_accum(
// CHECK-SAME:        %[[VAL_0:.*]] : ui2, "000000000000000", "001", "00000", "0010111")
// CHECK:          {
// CHECK:           %[[VAL_1:.*]] = hwarith.constant 0 : ui48

// CHECK:           coredsl.set @ACCUM[%[[VAL_0]] : ui2] = %[[VAL_1]] : ui48

// CHECK:           coredsl.end
// CHECK:         }

coredsl.isax "" {
    coredsl.register core_x @ACCUM[4] : ui48

    coredsl.instruction @reset_accum(%acc : ui2, "000000000000000", "001", "00000", "0010111") {
      %zero = hwarith.constant 0 : ui48

      coredsl.set @ACCUM[%acc : ui2] = %zero : ui48

      coredsl.end
    }
}

// -----

// CHECK:         coredsl.register local @ACCUM[4] = [1, 2, -3, -4] : si4

coredsl.isax "" {
    coredsl.register local @ACCUM[4] = [1, 2, -3, -4] : si4
}

// -----

coredsl.isax "" {
  // expected-error @+1 {{Const registers must be initialized}}
  coredsl.register local const @C1[5] : ui32
}
