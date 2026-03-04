#ifndef ADD_H
#define ADD_H

// =============================================================================
// Non-hierarchical versions — process entire tensors at once
// =============================================================================

// Both inputs same size: output[i] = a[i] + b[i]
void add_same(int size1, float* input_0, float* input_1, float* output_0);

// Bias: output[i] = a[i] + b[i % size2]
void add_bias(int size1, int size2, float* input_0, float* input_1, float* output_0);

// Scalar: output[i] = a[i] + b[0]
void add_scalar(int size1, int size2, float* input_0, float* input_1, float* output_0);

// Generic fallback
void add_generic(int size1, int size2, float* input_0, float* input_1, float* output_0);

// =============================================================================
// Hierarchical versions — process one element per firing
// =============================================================================

// Both inputs same size: output = a + b (one element)
void add_same_neuron(float* input_0, float* input_1, float* output_0);

// Bias: output = a + b (one element, b cycles via PREESM token rates)
void add_bias_neuron(float* input_0, float* input_1, float* output_0);

// Scalar: output = a + b (one element, b broadcast via PREESM)
void add_scalar_neuron(float* input_0, float* input_1, float* output_0);

// Generic fallback
void add_generic_neuron(float* input_0, float* input_1, float* output_0);

#endif
