// Will be linked to the @Add func defined in MLIR.
extern float Add(float, float);

float TestAdd(float a, float b) {
  return Add(a, b);
}
