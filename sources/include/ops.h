#ifndef OPS_H
#define OPS_H

// =============================================================================
// RELU
// =============================================================================
void relu(int size, float* input_0, float* output_0);
void relu_neuron(float* input_0, float* output_0);

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
// DROPOUT — at inference time this is just a passthrough
// =============================================================================
void dropout(int size, float* input_0, float* output_0);
void dropout_neuron(float* input_0, float* output_0);

// =============================================================================
// SOFTMAX — needs full vector, no neuron version
// =============================================================================
void softmax(int size, float* input_0, float* output_0);

// =============================================================================
// FLATTEN — just a copy, shape change is metadata
// =============================================================================
void flatten(int inputSize, int outputSize, float* input_0, float* output_0);
void flatten_neuron(float* input_0, float* output_0);

// =============================================================================
// CONCAT — concatenate two tensors along last axis
// =============================================================================
void concat(int size1, int size2, float* input_0, float* input_1, float* output_0);
void concat_neuron(float* input_0, float* input_1, float* output_0);

// =============================================================================
// MATMUL
// =============================================================================
void matmul(int M, int K, int N, float* input_0, float* input_1, float* output_0);
void matmul_neuron(int K, float* input_0, float* input_1, float* output_0);

// =============================================================================
// RESHAPE
// =============================================================================
void reshape(int inputSize, int outputSize, float* input_0, int* input_1, float* output_0);
void reshape_neuron(float* input_0, int* input_1, float* output_0);


#endif
