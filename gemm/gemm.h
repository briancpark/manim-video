#ifndef GEMM_H
#define GEMM_H

// Single-precision matrix multiply, row-major:  C = A * B
//   A : M x K   (lda = K)
//   B : K x N   (ldb = N)
//   C : M x N   (ldc = N)
// Each routine zeroes C internally, then accumulates.

void gemm_naive(const float *A, const float *B, float *C, int M, int N, int K);

// Cache-blocked, ikj loop order (scalar; relies on the compiler to vectorize
// the contiguous inner loop). Shows the win from tiling for the cache.
void gemm_blocked(const float *A, const float *B, float *C, int M, int N, int K);

// Cache-blocked + hand-written NEON 8x8 register microkernel.
void gemm_neon(const float *A, const float *B, float *C, int M, int N, int K);

#endif // GEMM_H
