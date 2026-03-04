#include "ops.h"
#include <math.h>
#include <string.h>
#include <float.h>

// =============================================================================
// RELU
// =============================================================================

void relu(int size, float* input_0, float* output_0) {
    for (int i = 0; i < size; i++)
        output_0[i] = input_0[i] > 0.0f ? input_0[i] : 0.0f;
}

void relu_neuron(float* input_0, float* output_0) {
    output_0[0] = input_0[0] > 0.0f ? input_0[0] : 0.0f;
}

// =============================================================================
// MATMUL
// =============================================================================

void matmul(int M, int K, int N, float* input_0, float* weight_0, float* output_0) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += input_0[m * K + k] * weight_0[k * N + n];
            output_0[m * N + n] = sum;
        }
}

// one row of input (K elements) + one col of weights (K elements) -> one scalar
void matmul_neuron(int K, float* input_0, float* weight_0, float* output_0) {
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += input_0[k] * weight_0[k];
    output_0[0] = sum;
}

// =============================================================================
// RESHAPE — shape change is just a copy, actual shape is handled by PREESM
// =============================================================================

void reshape(int inputSize, int outputSize, float* input_0, int* weight_0, float* output_0) {
    memcpy(output_0, input_0, inputSize * sizeof(float));
}

void reshape_neuron(float* input_0, int* weight_0, float* output_0) {
    output_0[0] = input_0[0];
}

void reshape_w(int outputSize, float* weight_0, int* weight_1, float* output_0) {
    memcpy(output_0, weight_0, outputSize * sizeof(float));
}

void reshape_w_neuron(float* weight_0, int* weight_1, float* output_0) {
    output_0[0] = weight_0[0];
}


