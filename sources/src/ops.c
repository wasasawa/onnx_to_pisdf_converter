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
// DROPOUT — passthrough at inference, training mask is ignored
// =============================================================================
void dropout(int size, float* input_0, float* output_0) {
    memcpy(output_0, input_0, size * sizeof(float));
}
void dropout_neuron(float* input_0, float* output_0) {
    output_0[0] = input_0[0];
}

// =============================================================================
// SOFTMAX — full vector only, no neuron version
// =============================================================================
void softmax(int size, float* input_0, float* output_0) {
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++)
        if (input_0[i] > max_val) max_val = input_0[i];

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output_0[i] = expf(input_0[i] - max_val);
        sum += output_0[i];
    }
    for (int i = 0; i < size; i++)
        output_0[i] /= sum;
}

// =============================================================================
// FLATTEN
// =============================================================================
void flatten(int inputSize, int outputSize, float* input_0, float* output_0) {
    memcpy(output_0, input_0, inputSize * sizeof(float));
}
void flatten_neuron(float* input_0, float* output_0) {
    output_0[0] = input_0[0];
}


// =============================================================================
// CONCAT
// =============================================================================
void concat(int size1, int size2, float* input_0, float* input_1, float* output_0) {
    memcpy(output_0,         input_0, size1 * sizeof(float));
    memcpy(output_0 + size1, input_1, size2 * sizeof(float));
}
void concat_neuron(float* input_0, float* input_1, float* output_0) {
    // one element from each stream interleaved — PREESM handles ordering via rates
    output_0[0] = input_0[0];
    output_0[1] = input_1[0];
}

// =============================================================================
// MATMUL
// =============================================================================
void matmul(int M, int K, int N, float* input_0, float* input_1, float* output_0) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += input_0[m * K + k] * input_1[k * N + n];
            output_0[m * N + n] = sum;
        }
}
// one row of A (K elements) x full B (K*N elements) -> one row of output (N elements)
void matmul_neuron(int K, int N, float* input_0, float* input_1, float* output_0) {
    for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += input_0[k] * input_1[k * N + n];
        output_0[n] = sum;
    }
}

// =============================================================================
// RESHAPE
// =============================================================================
void reshape(int inputSize, int outputSize, float* input_0, int* input_1, float* output_0) {
    memcpy(output_0, input_0, inputSize * sizeof(float));
}
void reshape_neuron(float* input_0, int* input_1, float* output_0) {
    output_0[0] = input_0[0];
}

