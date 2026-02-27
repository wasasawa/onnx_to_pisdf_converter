#ifndef ADD_H
#define ADD_H

#include <stddef.h>

/* Element-wise: A[i] + B[i], same shape */
void add_same(int size1, int size2, float* a, float* b, float* output);

/* Bias: A[i] + B[i % size2], size1 is multiple of size2 */
void add_bias(int size1, int size2, float* a, float* b, float* output);

/* Scalar: A[i] + B[0], broadcast single value */
void add_scalar(int size1, int size2, float* a, float* b, float* output);

/* Generic fallback */
void add_generic(int size1, int size2, float* a, float* b, float* output);

#endif
