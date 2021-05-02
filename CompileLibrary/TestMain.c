#include <stdio.h>

extern float TestAdd(float, float);

int main() {
  float a, b;

  a = 1.23f;
  b = 42.0f;

  printf("%f + %f = %f\n", a, b, TestAdd(a, b));

  return 0;
}
