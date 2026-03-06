#include "transpose.h"
#include <string.h>

/* ----------------------------------------------------------------------------
 * Generic N-dimensional transpose (up to 4-D).
 *
 * `ndim` is the true rank; extra slots in shape/perm are ignored.
 * The algorithm works in linearised coordinates:
 *   - strides are computed for both input and permuted output shapes.
 *   - For each input element, its output index is derived from permuted coords.
 * --------------------------------------------------------------------------*/
static void _transpose(
    int ndim,
    int in_dim_0, int in_dim_1, int in_dim_2, int in_dim_3,
    int perm_0,   int perm_1,   int perm_2,   int perm_3,
    const float* input_0, float* output_0
) {
    /* Build shape and permutation arrays — only the first `ndim` entries matter.
       Pad with 1 / identity for safety (though caller should also do this). */
    int shape[4] = { in_dim_0, in_dim_1, in_dim_2, in_dim_3 };
    int perm[4]  = { perm_0,   perm_1,   perm_2,   perm_3   };

    /* Output shape = input shape reordered by perm */
    int out_shape[4];
    for (int i = 0; i < ndim; i++)
        out_shape[i] = shape[perm[i]];

    /* Row-major (C-order) strides */
    int stride_in[4], stride_out[4];
    stride_in[ndim - 1]  = 1;
    stride_out[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        stride_in[i]  = stride_in[i  + 1] * shape[i  + 1];
        stride_out[i] = stride_out[i + 1] * out_shape[i + 1];
    }

    /* Total elements */
    int total = 1;
    for (int i = 0; i < ndim; i++) total *= shape[i];

    /* Iterate over every input element, compute its permuted output index */
    for (int lin = 0; lin < total; lin++) {

        /* Decompose linear index → per-axis coordinates (input order) */
        int coords[4];
        int tmp = lin;
        for (int i = ndim - 1; i >= 0; i--) {
            coords[i] = tmp % shape[i];
            tmp      /= shape[i];
        }

        /* Map to output linear index via permuted strides */
        int out_idx = 0;
        for (int i = 0; i < ndim; i++)
            out_idx += coords[perm[i]] * stride_out[i];

        output_0[out_idx] = input_0[lin];
    }
}

// =============================================================================
// Non-hierarchical
// =============================================================================

void transpose(
    int ndim,
    int in_dim_0, int in_dim_1, int in_dim_2, int in_dim_3,
    int perm_0,   int perm_1,   int perm_2,   int perm_3,
    float* input_0, float* output_0
) {
    _transpose(ndim,
               in_dim_0, in_dim_1, in_dim_2, in_dim_3,
               perm_0, perm_1, perm_2, perm_3,
               input_0, output_0);
}

// =============================================================================
// Hierarchical — identical; full tensor required for arbitrary permutation
// =============================================================================

void transpose_neuron(
    int ndim,
    int in_dim_0, int in_dim_1, int in_dim_2, int in_dim_3,
    int perm_0,   int perm_1,   int perm_2,   int perm_3,
    float* input_0, float* output_0
) {
    _transpose(ndim,
               in_dim_0, in_dim_1, in_dim_2, in_dim_3,
               perm_0, perm_1, perm_2, perm_3,
               input_0, output_0);
}

