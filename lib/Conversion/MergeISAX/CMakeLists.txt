add_shortnail_library(ShortnailMergeISAX
    MergeISAX.cpp

    DEPENDS
    MLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass

    ShortnailCoreDSL
)
