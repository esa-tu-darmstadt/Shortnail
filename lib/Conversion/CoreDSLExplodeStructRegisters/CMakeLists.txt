add_shortnail_library(ShortnailCoreDSLExplodeStructRegisters
    CoreDSLExplodeStructRegisters.cpp

    DEPENDS
    MLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass

    CIRCTHW

    ShortnailCoreDSL
)
