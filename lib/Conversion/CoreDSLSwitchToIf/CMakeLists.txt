add_shortnail_library(ShortnailCoreDSLSwitchToIf
    CoreDSLSwitchToIf.cpp

    DEPENDS
    MLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSCFUtils
    MLIRControlFlowToSCF

    CIRCTHW
    CIRCTHWArith
    CIRCTSupport

    ShortnailCoreDSL
)
