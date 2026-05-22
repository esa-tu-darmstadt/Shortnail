add_shortnail_library(ShortnailCoreDSLLegalizeCF
    CoreDSLLegalizeCF.cpp

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
