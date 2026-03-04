#ifndef TRANSPOSE_H
#define TRANSPOSE_H

// Non-hierarchical — full tensor reorder at once
void transpose(
    int size,
    int dim_0, int dim_1, int dim_2, int dim_3,
    int perm_0, int perm_1, int perm_2, int perm_3,
    float* input_0, float* output_0
);

// Hierarchical — same as non-hierarchical since we need the full tensor
// rates in transpose.pi are size/size (not 1/1) for this reason
void transpose_neuron(
    int size,
    int dim_0, int dim_1, int dim_2, int dim_3,
    int perm_0, int perm_1, int perm_2, int perm_3,
    float* input_0, float* output_0
);

#endif
