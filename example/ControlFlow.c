#include <stdlib.h>

struct TensorRecord { float *data; size_t size; };

typedef struct TensorRecord *Tensor;

extern Tensor Add(Tensor, Tensor);

Tensor Reduction(Tensor init, Tensor x, int N) {
  int i;
  Tensor y = init;
  for (i = 0; i < N; i ++)
    y = Add(y, x);
  return y;
}
