#include "transpose.h"

static void _transpose(
    int size,
    int dim_0, int dim_1, int dim_2, int dim_3,
    int perm_0, int perm_1, int perm_2, int perm_3,
    float* input_0, float* output_0
) {
    int shape[4]   = {dim_0,  dim_1,  dim_2,  dim_3};
    int perm[4]    = {perm_0, perm_1, perm_2, perm_3};
    int out_shape[4];
    int strides_in[4], strides_out[4];

    for (int i = 0; i < 4; i++)
        out_shape[i] = shape[perm[i]];

    strides_in[3] = 1;
    for (int i = 2; i >= 0; i--)
        strides_in[i] = strides_in[i+1] * shape[i+1];

    strides_out[3] = 1;
    for (int i = 2; i >= 0; i--)
        strides_out[i] = strides_out[i+1] * out_shape[i+1];

    for (int i0 = 0; i0 < shape[0]; i0++)
    for (int i1 = 0; i1 < shape[1]; i1++)
    for (int i2 = 0; i2 < shape[2]; i2++)
    for (int i3 = 0; i3 < shape[3]; i3++) {
        int coords[4] = {i0, i1, i2, i3};
        int idx_in    = i0*strides_in[0]  + i1*strides_in[1]
                      + i2*strides_in[2]  + i3*strides_in[3];
        int idx_out   = coords[perm[0]]*strides_out[0] + coords[perm[1]]*strides_out[1]
                      + coords[perm[2]]*strides_out[2] + coords[perm[3]]*strides_out[3];
        output_0[idx_out] = input_0[idx_in];
    }
}

void transpose(
    int size,
    int dim_0, int dim_1, int dim_2, int dim_3,
    int perm_0, int perm_1, int perm_2, int perm_3,
    float* input_0, float* output_0
) {
    _transpose(size, dim_0, dim_1, dim_2, dim_3,
               perm_0, perm_1, perm_2, perm_3,
               input_0, output_0);
}

// neuron version is identical since transpose needs the full tensor
void transpose_neuron(
    int size,
    int dim_0, int dim_1, int dim_2, int dim_3,
    int perm_0, int perm_1, int perm_2, int perm_3,
    float* input_0, float* output_0
) {
    _transpose(size, dim_0, dim_1, dim_2, dim_3,
               perm_0, perm_1, perm_2, perm_3,
               input_0, output_0);
}
