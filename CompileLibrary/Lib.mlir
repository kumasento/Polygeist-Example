func @Add(%a: f32, %b: f32) -> f32 {
  %c = addf %a, %b : f32
  return %c : f32
}
