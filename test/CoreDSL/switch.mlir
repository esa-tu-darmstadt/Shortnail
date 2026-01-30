// RUN: shortnail-opt %s | shortnail-opt | FileCheck %s

// TODO: rename
coredsl.isax "X_ALT_MAC" {
  coredsl.register core_x @X[32] : ui32
  coredsl.register local @ACC : ui64

  // CHECK: coredsl.instruction @SWITCH_TEST(
  coredsl.instruction @SWITCH_TEST("000000", %isSigned : ui1, %rs2 : ui5, %rs1 : ui5,
                               "000", %rd : ui5, "0101011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    %2 = coredsl.switch %1 : ui32 -> ui32
      case 0 {
        %2 = hwarith.constant 2 : ui32
        coredsl.yield %2 : ui32
      }
      default {
        %2 = hwarith.constant 3 : ui32
        coredsl.yield %2 : ui32
      }
    coredsl.end
  }
}
