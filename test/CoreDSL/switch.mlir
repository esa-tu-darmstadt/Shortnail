// RUN: shortnail-opt %s -canonicalize -split-input-file -verify-diagnostics | shortnail-opt | FileCheck %s

coredsl.isax "SWITCH_TEST" {
// CHECK: coredsl.register core_x @X[32] : ui32
// CHECK: coredsl.register local @ACC : ui64
  coredsl.register core_x @X[32] : ui32
  coredsl.register local @ACC : ui64

  // CHECK: coredsl.instruction @SWITCH_TEST(
  coredsl.instruction @SWITCH_TEST("000000", %isSigned : ui1, %rs2 : ui5, %rs1 : ui5,
                               "000", %rd : ui5, "0101011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    // CHECK: %[[VAL_1:.*]] = hwarith.constant 54 : ui32
    // CHECK: %[[VAL_0:.*]] = hwarith.constant 2 : ui32
    // CHECK: %[[VAL_2:.*]] = hwarith.constant 3 : ui32
    // CHECK: %[[LOADED_0:.*]] = coredsl.get @X[%rs1 : ui5] : ui32
    // CHECK: %[[LOADED_1:.*]] = coredsl.get @X[%rs2 : ui5] : ui32
    // CHECK: %[[RES:.*]] = coredsl.switch %[[COND:.*]] : ui32 -> ui32
    // CHECK: case 0 {
    // CHECK: coredsl.yield %[[VAL_0]] : ui32
    // CHECK: }
    // CHECK: case 2 {
    // CHECK: coredsl.yield %[[VAL_1]] : ui32
    // CHECK: }
    // CHECK: default {
    // CHECK: coredsl.yield %[[VAL_2]] : ui32
    // CHECK: }
    // CHECK: coredsl.set @X[%rs1 : ui5] = %[[RES]] : ui32
    %2 = coredsl.switch %1 : ui32 -> ui32
      case 0 {
        %2 = hwarith.constant 2 : ui32
        coredsl.yield %2 : ui32
      }
      case 2 {
        %2 = hwarith.constant 54 : ui32
        coredsl.yield %2 : ui32
      }
      default {
        %2 = hwarith.constant 3 : ui32
        coredsl.yield %2 : ui32
      }
    coredsl.set @X[%rs1 : ui5] = %2 : ui32
    coredsl.end
  }

  coredsl.instruction @NO_ERROR_SIGNED_INT("000000", %isSigned : ui1, %rs2 : ui5, %rs1 : ui5,
                               "000", %rd : ui5, "0101011") {    
    // CHECK: %[[RES_0:.*]] = hwarith.constant 54 : ui32
    // CHECK: %[[RES_1:.*]] = hwarith.constant 2 : ui32
    // CHECK: %[[RES_2:.*]] = hwarith.constant 3 : ui32
    // CHECK: %[[READ_VAL:.*]] = coredsl.get @X[%rs1 : ui5] : ui32
    // CHECK: %[[COND_UI32:.*]] = coredsl.get @X[%rs2 : ui5] : ui32
    // CHECK: %[[COND:.*]] = coredsl.cast %[[COND_UI32]] : ui32 to si8
    // CHECK: %[[COND_RES:.*]] = coredsl.switch %[[COND]] : si8 -> ui32
    // CHECK:   case 0 {
    // CHECK:     coredsl.yield %[[RES_1]] : ui32
    // CHECK:   }
    // CHECK:   case -1 {
    // CHECK:     coredsl.yield %[[RES_0]] : ui32
    // CHECK:   }
    // CHECK:   default {
    // CHECK:     coredsl.yield %[[RES_2]] : ui32
    // CHECK:   }
    // CHECK: coredsl.set @X[%rs1 : ui5] = %[[COND_RES]] : ui32

    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    %2 = coredsl.cast %1 : ui32 to si8
    %3 = coredsl.switch %2 : si8 -> ui32
      case 0 {
        %3 = hwarith.constant 2 : ui32
        coredsl.yield %3 : ui32
      }
      case -1 {
        %3 = hwarith.constant 54 : ui32
        coredsl.yield %3 : ui32
      }
      default {
        %3 = hwarith.constant 3 : ui32
        coredsl.yield %3 : ui32
      }
    coredsl.set @X[%rs1 : ui5] = %3 : ui32
    coredsl.end
  }

  coredsl.instruction @CONST_CASE_FOLD("000000", %isSigned : ui1, %rs2 : ui5, %rs1 : ui5,
                               "000", %rd : ui5, "0101011") {
    // CHECK: %[[RES:.*]] = hwarith.constant 54 : ui32
    // CHECK: %[[LOADED_0:.*]] = coredsl.get @X[%rs1 : ui5] : ui32
    // CHECK: %[[LOADED_1:.*]] = coredsl.get @X[%rs2 : ui5] : ui32
    // CHECK: coredsl.set @X[%rs1 : ui5] = %[[RES]] : ui32
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    %2 = hwarith.constant -1 : si8
    %3 = coredsl.switch %2 : si8 -> ui32
      case 0 {
        %3 = hwarith.constant 2 : ui32
        coredsl.yield %3 : ui32
      }
      case -1 {
        %3 = hwarith.constant 54 : ui32
        coredsl.yield %3 : ui32
      }
      default {
        %3 = hwarith.constant 3 : ui32
        coredsl.yield %3 : ui32
      }
    coredsl.set @X[%rs1 : ui5] = %3 : ui32
    coredsl.end
  }

  coredsl.instruction @CONST_CASE_FOLD_LARGER_THAN_64BIT("000000", %isSigned : ui1, %rs2 : ui5, %rs1 : ui5,
                               "000", %rd : ui5, "0101011") {
    // CHECK: %[[RES:.*]] = hwarith.constant 54 : ui32
    // CHECK: coredsl.set @X[%rs1 : ui5] = %[[RES]] : ui32
    %2 = hwarith.constant 36893488147419103232 : ui128
    %3 = coredsl.switch %2 : ui128 -> ui32
      case 0 {
        %3 = hwarith.constant 2 : ui32
        coredsl.yield %3 : ui32
      }
      case 36893488147419103232 {
        %3 = hwarith.constant 54 : ui32
        coredsl.yield %3 : ui32
      }
      default {
        %3 = hwarith.constant 3 : ui32
        coredsl.yield %3 : ui32
      }
    coredsl.set @X[%rs1 : ui5] = %3 : ui32
    coredsl.end
  }
}
