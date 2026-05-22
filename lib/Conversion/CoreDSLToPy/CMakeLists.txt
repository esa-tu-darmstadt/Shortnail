add_shortnail_library(ShortnailCoreDSLToPy
    CoreDSLToPy.cpp

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
)
