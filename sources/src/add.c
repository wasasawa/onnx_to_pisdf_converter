#include "add.h"

// =============================================================================
// Non-hierarchical versions
// =============================================================================

void add_same(int size1, float* input_0, float* input_1, float* output_0) {
    for (int i = 0; i < size1; i++)
        output_0[i] = input_0[i] + input_1[i];
}

void add_bias(int size1, int size2, float* input_0, float* input_1, float* output_0) {
    for (int i = 0; i < size1; i++)
        output_0[i] = input_0[i] + input_1[i % size2];
}

void add_scalar(int size1, int size2, float* input_0, float* input_1, float* output_0) {
    float scalar = (size2 == 1) ? input_1[0] : input_0[0];
    float* tensor = (size2 == 1) ? input_0 : input_1;
    int size = (size2 == 1) ? size1 : size2;
    for (int i = 0; i < size; i++)
        output_0[i] = tensor[i] + scalar;
}

void add_generic(int size1, int size2, float* input_0, float* input_1, float* output_0) {
    if (size2 == 1 || size1 == 1) {
        add_scalar(size1, size2, input_0, input_1, output_0);
    } else if (size1 % size2 == 0) {
        add_bias(size1, size2, input_0, input_1, output_0);
    } else if (size2 % size1 == 0) {
        add_bias(size2, size1, input_1, input_0, output_0);
    } else {
        // sizes don't match any known pattern — add up to min size
        int n = size1 < size2 ? size1 : size2;
        for (int i = 0; i < n; i++)
            output_0[i] = input_0[i] + input_1[i];
    }
}

// =============================================================================
// Hierarchical versions — called once per element by PREESM
// =============================================================================

void add_same_neuron(int size1, int size2, float* input_0, float* input_1, float* output_0) {
    for (int i = 0; i < size1; i++)
        output_0[i] = input_0[i] + input_1[i];
}

void add_bias_neuron(float* input_0, float* input_1, float* output_0) { 
    output_0[0] = input_0[0] + input_1[0];
}

void add_scalar_neuron(float* input_0, float* input_1, float* output_0) {
    output_0[0] = input_0[0] + input_1[0];
}

void add_generic_neuron(float* input_0, float* input_1, float* output_0) {
    output_0[0] = input_0[0] + input_1[0];
}
