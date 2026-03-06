#ifndef OPS_H
#define OPS_H

// =============================================================================
// RELU
// =============================================================================
void relu(int size, float* input_0, float* output_0);
void relu_neuron(int size, float* input_0, float* output_0);

// =============================================================================
// SIGMOID
// =============================================================================
void sigmoid(int size, float* input_0, float* output_0);
void sigmoid_neuron(float* input_0, float* output_0);

// =============================================================================
// TANH
// =============================================================================
void tanh_op(int size, float* input_0, float* output_0);
void tanh_op_neuron(float* input_0, float* output_0);

// =============================================================================
// DROPOUT — passthrough at inference time
// =============================================================================
void dropout(int size, float* input_0, float* output_0);
void dropout_neuron(float* input_0, float* output_0);

// =============================================================================
// SOFTMAX — needs full vector, no neuron version
// =============================================================================
void softmax(int size, int outerSize, float* input_0, float* output_0);

// =============================================================================
// FLATTEN — shape change is metadata; data is a plain copy
// =============================================================================
void flatten(int inputSize, int outputSize, float* input_0, float* output_0);
void flatten_neuron(int inputSize, int outputSize, float* input_0, float* output_0);

// =============================================================================
// CONCAT — concatenate two tensors along last axis
// =============================================================================
void concat(int size1, int size2, float* input_0, float* input_1, float* output_0);
void concat_neuron(int size1, int size2,float* input_0, float* input_1, float* output_0);

// =============================================================================
// MATMUL
//   A [M × K]  (data input)
//   B [K × N]  (weight)
//   Y [M × N]
// =============================================================================
void matmul(int M, int K, int N, float* input_0, float* input_1, float* output_0);
/* Neuron: one row of A (K elements) × full B (K×N) → one row of Y (N elements) */
void matmul_neuron(int M, int K, int N, float* input_0, float* input_1, float* output_0);

// =============================================================================
// RESHAPE — data is a plain copy; shape tensor is a weight (int)
// =============================================================================
void reshape(int inputSize, int outputSize, float* input_0, int* input_1, float* output_0);
void reshape_neuron(int inputSize, int outputSize, float* input_0, int* input_1, float* output_0);


#endif
