add_shortnail_library(ShortnailCoreDSLLiftCFToSCF
    CoreDSLLiftCFToSCF.cpp

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
