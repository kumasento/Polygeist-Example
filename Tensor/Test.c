#include <stdio.h>

#define Tensor void *

Tensor FromData();
Tensor Add(Tensor, Tensor);
void PreventDCE(Tensor);

int main() {
  Tensor A = FromData();
  Tensor B = Add(A, A);
  PreventDCE(B);
  return 0;
}
