/**
 * gemm.cpp  –  ONNX "Gemm" operator
 *
 * ONNX spec (opset 13+):
 *   Y = alpha * op(A) * op(B) + beta * C
 *   where op(X) = X if transX=0 else X^T
 *
 *   Inputs : A [M, K]  –  data input (live activation)
 *            B [K, N] or [N, K] if transB=1  –  weight matrix (initializer)
 *            C [N] or [M, N]                 –  bias   (optional initializer)
 *   Output : Y [M, N]
 *
 * Parameter order (PiSDF convention):
 *   config params (int) : M, K, N, transA, transB, sizeC
 *   config params (float): alpha, beta
 *   data inputs    : input_0  (A)
 *   weight inputs  : input_1 (B),  input_2 (C)
 *   output         : output_0 (Y)
 *
 * Bugs fixed vs original:
 *   - Removed internal file loading for B and C (arrive as FIFO pointers).
 *   - Removed fixed-point alpha/beta encoding (use float directly).
 *   - Removed duplicated op_gemm() function appended at the bottom.
 *   - sizeC=0 safely skips C addition when C is absent.
 *
 * C broadcast rules (per ONNX spec):
 *   C shape [N]     → broadcast over rows (add C[j] to every row)
 *   C shape [M, N]  → full matrix add
 *   C shape [1]     → scalar broadcast (edge case)
 */

#include <stdio.h>
#include "common.h"
#include "gemm.h"

void gemm(
    IN int    M,
    IN int    K,
    IN int    transA,
    IN int  alpha,
    IN int    N, 
    IN int    transB,
    IN int  beta,
    IN int    sizeC,
    IN float *input_0,  /* A [M×K] or [K×M] if transA=1 */
    IN float *input_1, /* B [K×N] or [N×K] if transB=1 */
    IN float *input_2, /* C – may be NULL when sizeC=0 */
    OUT float *output_0)/* Y [M×N] */
{
    float alpha_f = (float)alpha;   
    float beta_f  = (float)beta;
#ifdef VERBOSE_GEMM
    printf("Gemm: Y = %.3f * op(A)[%d×%d] * op(B)[%d×%d] + %.3f * C  "
           "(transA=%d transB=%d sizeC=%d)\n",
           alpha_f, M, K, K, N, beta_f, transA, transB, sizeC);
#endif
    // printf("[GEMM DEBUG] alpha raw bits=0x%08X, as_float=%e, as_int=%d\n",
    //     *(unsigned int*)&alpha, alpha, *(int*)&alpha);
    // printf("[GEMM DEBUG] beta raw bits=0x%08X, as_float=%e, as_int=%d\n",
    //     *(unsigned int*)&beta, beta, *(int*)&beta);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
             
            float sum = 0.0f;

            for (int k = 0; k < K; k++) {
                float a = transA ? input_0 [k * M + i] : input_0 [i * K + k];
                float b = transB ? input_1[j * K + k] : input_1[k * N + j];
                sum += a * b;
            }

            sum *= alpha_f;

            /* Add beta * C (with broadcasting) */
            if (input_2 && sizeC > 0) {
                float c_val;
                if (sizeC == 1) {
                    /* scalar */
                    c_val = input_2[0];
                } else if (sizeC == N) {
                    /* row-vector broadcast [N] → each row */
                    c_val = input_2[j];
                } else if (sizeC == M * N) {
                    /* full matrix [M, N] */
                    c_val = input_2[i * N + j];
                } else {
                    /* Unexpected shape – best-effort: treat as row-vector,
                       clamp index to avoid OOB.                             */
                    c_val = input_2[j % sizeC];
                }
                sum += beta_f * c_val;
            }

            output_0[i * N + j] = sum;
            // if (i == 0) { // Batch 0
            //     float feature_sum = 0;
            //     for(int k=0; k<5; k++) feature_sum += input_0[k]; 
            //     printf("DEBUG GEMM Class %d: First 5 feat sum=%f | Bias=%f | Result=%f\n", 
            //             j, feature_sum, (input_2 ? input_2[j] : 0.0f), output_0[j]);
            // }  
        }
    }
}



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
    OUT float *output_0){
        gemm(M,K,transA,alpha,N,transB,beta,sizeC,input_0,input_1,input_2,output_0);
    }