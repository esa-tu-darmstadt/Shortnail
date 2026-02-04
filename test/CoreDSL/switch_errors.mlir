// RUN: shortnail-opt %s -split-input-file -verify-diagnostics

coredsl.isax "SWITCH_ERRORS" {
  coredsl.register core_x @X[32] : ui32
  coredsl.register local @ACC : ui64
  
  coredsl.instruction @ERROR_VALUE_TOO_LARGE("000000", %isSigned : ui1, %rs2 : ui5, %rs1 : ui5,
                               "000", %rd : ui5, "0101011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    %2 = coredsl.cast %1 : ui32 to ui8
    // expected-error @+1 {{'coredsl.switch' op expects case value to be representable by ui8 but got 50000}}
    %3 = coredsl.switch %2 : ui8 -> ui32
      case 0 {
        %3 = hwarith.constant 2 : ui32
        coredsl.yield %3 : ui32
      }
      case 50000 {
        %3 = hwarith.constant 54 : ui32
        coredsl.yield %3 : ui32
      }
      default {
        %3 = hwarith.constant 3 : ui32
        coredsl.yield %3 : ui32
      }
    coredsl.end
  }
  coredsl.instruction @ERROR_NEGATIVE_FOR_UNSIGNED("000000", %isSigned : ui1, %rs2 : ui5, %rs1 : ui5,
                               "000", %rd : ui5, "0101011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    %2 = coredsl.cast %1 : ui32 to ui8
    // expected-error @+1 {{'coredsl.switch' op expects case value to be representable by ui8 but got -1}}
    %3 = coredsl.switch %2 : ui8 -> ui32
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
    coredsl.end
  }

  coredsl.instruction @ERROR_SIGNED_INT_TOO_LARGE("000000", %isSigned : ui1, %rs2 : ui5, %rs1 : ui5,
                               "000", %rd : ui5, "0101011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    %2 = coredsl.cast %1 : ui32 to si8
    // expected-error @+1 {{'coredsl.switch' op expects case value to be representable by si8 but got 150}}
    %3 = coredsl.switch %2 : si8 -> ui32
      case 0 {
        %3 = hwarith.constant 2 : ui32
        coredsl.yield %3 : ui32
      }
      // should error
      case 150 {
        %3 = hwarith.constant 54 : ui32
        coredsl.yield %3 : ui32
      }
      default {
        %3 = hwarith.constant 3 : ui32
        coredsl.yield %3 : ui32
      }
    coredsl.end
  }
  coredsl.instruction @ERROR_SIGNED_INT_NEGATIVE_LARGE("000000", %isSigned : ui1, %rs2 : ui5, %rs1 : ui5,
                               "000", %rd : ui5, "0101011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    %2 = coredsl.cast %1 : ui32 to ui128
    // expected-error @+1 {{'coredsl.switch' op expects case value to be representable by ui128 but got -18446744073709551616}}
    %3 = coredsl.switch %2 : ui128 -> ui32
      case 0 {
        %3 = hwarith.constant 2 : ui32
        coredsl.yield %3 : ui32
      }
      // Should be a 65 bit signed integer
      case -18446744073709551616 {
        %3 = hwarith.constant 54 : ui32
        coredsl.yield %3 : ui32
      }
      default {
        %3 = hwarith.constant 3 : ui32
        coredsl.yield %3 : ui32
      }
    coredsl.end
  }
}
