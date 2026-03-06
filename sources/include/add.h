#ifndef ADD_H
#define ADD_H

// =============================================================================
// Non-hierarchical — process entire tensors at once
// =============================================================================

/* Both inputs same size: output[i] = a[i] + b[i] */
void add_same(int size1, float* input_0, float* input_1, float* output_0);

/* Bias: output[i] = a[i] + b[i % size2]  (size1 is a multiple of size2) */
void add_bias(int size1, int size2, float* input_0, float* input_1, float* output_0);

/* Scalar: one operand is a single element broadcast over the other */
void add_scalar(int size1, int size2, float* input_0, float* input_1, float* output_0);

/* Generic fallback — dispatches to one of the above at runtime */
void add_generic(int size1, int size2, float* input_0, float* input_1, float* output_0);

// =============================================================================
// Hierarchical — one element per firing; PREESM controls token rates
// =============================================================================
void add_same_neuron  (int size1, int size2, float* input_0, float* input_1, float* output_0);
void add_bias_neuron  (int size1, int size2, float* input_0, float* input_1, float* output_0);
void add_scalar_neuron(int size1, int size2, float* input_0, float* input_1, float* output_0);
void add_generic_neuron(int size1, int size2, float* input_0, float* input_1, float* output_0);

#endif /* ADD_H */
