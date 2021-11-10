set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(runner
  runner.cc
)
llvm_update_compile_flags(runner)
target_link_libraries(runner
  PRIVATE
  MLIRStandard
  MLIRLLVMIR
  MLIRMath
  MLIRLoopAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR

  PgexTransforms  
)
