add_mlir_library(PgexTransforms
  MapToDialects.cc
  SimplifyDataflow.cc
  Ptr2Tsr.cc

  ADDITIONAL_HEADER_DIRS
  "${PROJ_MAIN_INCLUDE_DIR}/Transforms"  

  DEPENDS
  PgexMLIRTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRIR
  MLIRStandard
  MLIRSupport
  MLIRAffineToStandard
  MLIRTosa
)
