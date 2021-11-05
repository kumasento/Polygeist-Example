#include <stdlib.h>

struct TensorRecord { float *data; size_t size; };

typedef struct TensorRecord *Tensor;

extern Tensor Add(Tensor x, Tensor y);
extern Tensor Mul(Tensor x, Tensor y);

Tensor Compute(Tensor x, Tensor y) {
  return Add(Mul(x, x), Mul(y, y));
}
