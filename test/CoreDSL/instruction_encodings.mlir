// RUN: shortnail-opt %s -split-input-file -verify-diagnostics | shortnail-opt -verify-diagnostics | FileCheck %s

coredsl.isax "" {
    // expected-error @+1 {{encoding argument type must be an arbitrary precision integer with signedness semantics}}
    coredsl.instruction @bad_encoding_1(%acc: i2, "000000000000000", "000", "00000", "0000000") {
      coredsl.end
    }
}

// -----

coredsl.isax "" {
    // expected-error @+1 {{invalid character in encoding, only '0' and '1' are valid}}
    coredsl.instruction @bad_encoding_2(%acc: ui2, "%wtf", "000000000000000", "000", "00000", "0000000") {
      coredsl.end
    }
}

// -----

// CHECK-LABEL:   coredsl.instruction @preserve_names(
// CHECK-SAME:         %acc : ui2, "000000000000000", %rd : ui3, %rs : ui4, "0", "0000000")
// CHECK:         {
// CHECK:           %[[_VAR_3:.*]] = hwarith.add %rd, %acc : (ui3, ui2) -> ui4
// CHECK:           coredsl.end
// CHECK:         }

coredsl.isax "" {
    coredsl.instruction @preserve_names(%acc: ui2, "000000000000000", %rd: ui3, %rs: ui4, "0", "0000000") {
      %0 = hwarith.add %rd, %acc : (ui3, ui2) -> ui4
      coredsl.end
    }
}
