// RUN: shortnail-opt %s | shortnail-opt | FileCheck %s

coredsl.isax "X_ALT_MAC" {
  coredsl.register core_x @X[32] : ui32
  coredsl.register local @ACC : ui64

  // CHECK: coredsl.instruction @ALT_MAC(
  coredsl.instruction @ALT_MAC("000000", %isSigned : ui1, %rs2 : ui5, %rs1 : ui5,
                               "000", %rd : ui5, "0101011") {
    %0 = coredsl.get @X[%rs1 : ui5] : ui32
    %1 = coredsl.get @X[%rs2 : ui5] : ui32
    %cc = coredsl.cast %isSigned : ui1 to i1
    %newAcc = scf.if %cc -> ui64 {
      %2 = coredsl.cast %0 : ui32 to si32
      %3 = coredsl.cast %1 : ui32 to si32
      %4 = hwarith.mul %2, %3 : (si32, si32) -> si64
      %5 = coredsl.get @ACC : ui64
      %6 = coredsl.cast %5 : ui64 to si64
      %7 = hwarith.add %4, %6 : (si64, si64) -> si65
      %8 = coredsl.bitextract %7[63:0] : (si65) -> ui64
      scf.yield %8 : ui64
    } else {
      %2 = hwarith.mul %0, %1 : (ui32, ui32) -> ui64
      %3 = coredsl.get @ACC : ui64
      %4 = hwarith.add %2, %3 : (ui64, ui64) -> ui65
      %5 = coredsl.bitextract %4[63:0] : (ui65) -> ui64
      scf.yield %5 : ui64
    }
    coredsl.set @ACC = %newAcc : ui64
    coredsl.end
  }
}
