#ifndef TRANSPOSE_H
#define TRANSPOSE_H

/* Parameter order (PREESM/converter convention):
 *   ndim                           ← actual number of dimensions (1–4)
 *   in_dim_0, in_dim_1, in_dim_2, in_dim_3   ← input shape (unused dims = 1)
 *   perm_0,   perm_1,   perm_2,   perm_3     ← permutation (unused = identity)
 */

/* Non-hierarchical — full tensor reorder at once */
void transpose(
    int ndim,
    int in_dim_0, int in_dim_1, int in_dim_2, int in_dim_3,
    int perm_0,   int perm_1,   int perm_2,   int perm_3,
    float* input_0, float* output_0
);

/* Hierarchical — identical to non-hierarchical; rates are size/size, not 1/1,
   because the full tensor is required for an arbitrary permutation. */
void transpose_neuron(
    int ndim,
    int in_dim_0, int in_dim_1, int in_dim_2, int in_dim_3,
    int perm_0,   int perm_1,   int perm_2,   int perm_3,
    float* input_0, float* output_0
);

#endif
