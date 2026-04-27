add_shortnail_library(ShortnailCoreDSLSwitchToIf
    CoreDSLSwitchToIf.cpp

    DEPENDS
    MLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRSCFUtils

    CIRCTHW
    CIRCTHWArith
    CIRCTSupport

    ShortnailCoreDSL
    ShortnailCoreDSLLiftCFToSCF
)
