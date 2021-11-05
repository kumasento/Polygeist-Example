module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @TestAdd(%arg0: f32, %arg1: f32) -> f32 {
    %0 = call @Add(%arg0, %arg1) : (f32, f32) -> f32
    return %0 : f32
  }
  func private @Add(f32, f32) -> f32
}
