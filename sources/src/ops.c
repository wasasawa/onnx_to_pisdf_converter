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
void relu_neuron(int size, float* input_0, float* output_0) {
    for (int i = 0; i < size; i++)
        output_0[i] = input_0[i] > 0.0f ? input_0[i] : 0.0f;
}

// =============================================================================
// SIGMOID
// =============================================================================
void sigmoid(int size, float* input_0, float* output_0) {
    for (int i = 0; i < size; i++)
        output_0[i] = 1.0f / (1.0f + expf(-input_0[i]));
}
void sigmoid_neuron(float* input_0, float* output_0) {
    output_0[0] = 1.0f / (1.0f + expf(-input_0[0]));
}

// =============================================================================
// TANH — named tanh_op to avoid conflict with C standard tanh()
// =============================================================================
void tanh_op(int size, float* input_0, float* output_0) {
    for (int i = 0; i < size; i++)
        output_0[i] = tanhf(input_0[i]);
}
void tanh_op_neuron(float* input_0, float* output_0) {
    output_0[0] = tanhf(input_0[0]);
}

// =============================================================================
// DROPOUT — passthrough at inference; training mask is ignored
// =============================================================================
void dropout(int size, float* input_0, float* output_0) {
    memcpy(output_0, input_0, size * sizeof(float));
}
void dropout_neuron(float* input_0, float* output_0) {
    output_0[0] = input_0[0];
}

// =============================================================================
// SOFTMAX
//   size     : total elements  (= outerSize * innerSize)
//   outerSize: batch / independent softmax vectors
//   innerSize = size / outerSize
//
// Numerically stable: subtract per-vector max before exponentiation.
// No neuron version — requires the complete inner vector.
// =============================================================================
void softmax(int size, int outerSize, float* input_0, float* output_0) {
    int inner = size / outerSize;

    for (int b = 0; b < outerSize; b++) {
        const float* in  = input_0  + b * inner;
        float*       out = output_0 + b * inner;

        /* 1. find max for numerical stability */
        float max_val = -FLT_MAX;
        for (int i = 0; i < inner; i++)
            if (in[i] > max_val) max_val = in[i];

        /* 2. compute shifted exp and accumulate */
        float sum = 0.0f;
        for (int i = 0; i < inner; i++) {
            out[i] = expf(in[i] - max_val);
            sum   += out[i];
        }

        /* 3. normalise */
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < inner; i++)
            out[i] *= inv_sum;
    }
}

// =============================================================================
// FLATTEN — shape change is metadata; data is a plain copy
// =============================================================================
void flatten(int inputSize, int outputSize, float* input_0, float* output_0) {
    memcpy(output_0, input_0, inputSize * sizeof(float));
}
void flatten_neuron(int inputSize, int outputSize, float* input_0, float* output_0) {
    memcpy(output_0, input_0, inputSize * sizeof(float));
}

// =============================================================================
// CONCAT — concatenate two tensors along the last axis
// =============================================================================
void concat(int size1, int size2, float* input_0, float* input_1, float* output_0) {
    memcpy(output_0,         input_0, size1 * sizeof(float));
    memcpy(output_0 + size1, input_1, size2 * sizeof(float));
}
/* Neuron: one element from each stream — PREESM handles ordering via rates */
void concat_neuron(int size1, int size2,float* input_0, float* input_1, float* output_0) {
    output_0[0] = input_0[0];
    output_0[1] = input_1[0];
}

// =============================================================================
// MATMUL   A[M×K] × B[K×N] → Y[M×N]
// =============================================================================
void matmul(int M, int K, int N, float* input_0, float* input_1, float* output_0) {
    for (int m = 0; m < M; m++) {
        const float* a_row = input_0 + m * K;
        float*       y_row = output_0 + m * N;
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += a_row[k] * input_1[k * N + n];
            y_row[n] = sum;
        }
    }
}
/* Neuron: one row of A (K elements) × full B (K×N) → one row of Y (N elements) */
void matmul_neuron(int M, int K, int N, float* input_0, float* input_1, float* output_0) {
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += input_0[k] * input_1[k * N + n];
        output_0[n] = sum;
    }
}

// =============================================================================
// RESHAPE — plain copy; shape tensor (int*) is a weight used only by PREESM
// =============================================================================
void reshape(int inputSize, int outputSize, float* input_0, int* input_1, float* output_0) {
    memcpy(output_0, input_0, inputSize * sizeof(float));
}
void reshape_neuron(float* input_0, int* input_1, float* output_0) {
    output_0[0] = input_0[0];
}
