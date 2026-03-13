#pragma once
#include <cblas.h>

namespace rt {

// ---------------------------------------------------------------------------
// Thin CBLAS wrapper so generated actor code stays readable.
// C = alpha * op(A) * op(B) + beta * C   (row-major)
// ---------------------------------------------------------------------------
inline void sgemm(
    bool         transA, bool         transB,
    int          M,      int          N,     int   K,
    float        alpha,
    const float* A,      int          lda,
    const float* B,      int          ldb,
    float        beta,
    float*       C,      int          ldc)
{
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                M, N, K,
                alpha, A, lda,
                       B, ldb,
                beta,  C, ldc);
}

} // namespace rt
