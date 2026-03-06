#ifndef GEMM_H
#define GEMM_H
/*
 * gemm.h  –  ONNX "Gemm" operator
 *
 * Y = alpha * op(A) * op(B) + beta * C
 *   op(X) = X if transX=0, X^T if transX=1
 *
 * A: [M, K]  (live data activation)
 * B: [K, N] or [N, K] when transB=1  (weight initializer)
 * C: [N] or [M,N]                    (bias  initializer, optional)
 * Y: [M, N]
 *
 * C broadcasting:
 *   sizeC == 0   → no bias (weight_1 is unused)
 *   sizeC == N   → row-vector broadcast
 *   sizeC == M*N → full matrix
 *   sizeC == 1   → scalar
 *
 * ONNX spec: opset 13+
 */

#include "common.h"
#include "preesm.h"
/**
 * gemm
 *
 * Config params:
 * @param M       (int)   Rows    of op(A)
 * @param K       (int)   Columns of op(A) / Rows of op(B)
 * @param N       (int)   Columns of op(B)
 * @param transA  (int)   0 = no transpose, 1 = transpose A
 * @param transB  (int)   0 = no transpose, 1 = transpose B
 * @param sizeC   (int)   Number of elements in C (0 if absent)
 * @param alpha   (float) Scalar multiplier for A*B  (default 1.0)
 * @param beta    (float) Scalar multiplier for C    (default 1.0)
 *
 * Port tokens:
 * @param input_0   A data     [M × K]           [data   IN]
 * @param input_1  B weights  [K × N]            [weight IN]
 * @param input_2  C bias     [sizeC]            [weight IN]  (may be NULL)
 * @param output_0  Y result   [M × N]            [data   OUT]
 */
void gemm(
    IN int    M,
    IN int    K,
    IN int    transA,
    IN int  alpha,
    IN int    N, 
    IN int    transB,
    IN int  beta,
    IN int    sizeC,
    IN float *input_0,
    IN float *input_1,
    IN float *input_2,
    OUT float *output_0);


void gemm_neuron(
    IN int    M,
    IN int    K,
    IN int    transA,
    IN int  alpha,
    IN int    N, 
    IN int    transB,
    IN int  beta,
    IN int    sizeC,
    IN float *input_0,
    IN float *input_1,
    IN float *input_2,
    OUT float *output_0);

#endif
