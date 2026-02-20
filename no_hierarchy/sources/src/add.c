#include "add.h"

/* A[i] + B[i] — both same size */
void add_same(int size1, int size2, float* a, float* b, float* output) {
    for (int i = 0; i < size1; i++)
        output[i] = a[i] + b[i];
}

/* A[i] + B[i % size2] — B is repeated across A */
/* e.g. A is [N, C, H, W] and B is [C] (bias per channel) */
void add_bias(int size1, int size2, float* a, float* b, float* output) {
    for (int i = 0; i < size1; i++)
        output[i] = a[i] + b[i % size2];
}

/* A[i] + B[0] — B is a single scalar broadcast to all */
void add_scalar(int size1, int size2, float* a, float* b, float* output) {
    float scalar = b[0];
    for (int i = 0; i < size1; i++)
        output[i] = a[i] + scalar;
}

/* Generic: try bias pattern, fall back to scalar */
void add_generic(int size1, int size2, float* a, float* b, float* output) {
    if (size2 == 1) {
        add_scalar(size1, size2, a, b, output);
    } else if (size1 % size2 == 0) {
        add_bias(size1, size2, a, b, output);
    } else {
        /* sizes don't match any known pattern — just do element-wise up to min */
        int n = size1 < size2 ? size1 : size2;
        for (int i = 0; i < n; i++)
            output[i] = a[i] + b[i];
    }
}
