#include "gemm.h"
#include <string.h>

// Cache blocking + ikj loop order.
//
// Two independent ideas, both visible here:
//  1) ikj order: the innermost loop over j makes C[i][j] and B[k][j] both
//     STRIDE-1 and contiguous. A[i][k] is loop-invariant in j, so it's hoisted
//     to a scalar broadcast. The compiler auto-vectorizes this inner loop.
//  2) Tiling: we process the matrices in blocks (MC x KC x NC) chosen so the
//     active slices of B and C stay resident in cache and get REUSED across the
//     block, instead of streaming the whole matrix from DRAM on every pass.
#ifndef MC
#define MC 64
#endif
#ifndef KC
#define KC 256
#endif
#ifndef NC
#define NC 512
#endif

void gemm_blocked(const float *A, const float *B, float *C, int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int jc = 0; jc < N; jc += NC) {
        int jmax = jc + NC < N ? jc + NC : N;
        for (int kc = 0; kc < K; kc += KC) {
            int kmax = kc + KC < K ? kc + KC : K;
            for (int ic = 0; ic < M; ic += MC) {
                int imax = ic + MC < M ? ic + MC : M;
                // micro-block: C[i, jc:jmax] += A[i, kc:kmax] * B[kc:kmax, jc:jmax]
                for (int i = ic; i < imax; i++) {
                    for (int k = kc; k < kmax; k++) {
                        float a = A[(size_t)i * K + k];          // broadcast scalar
                        const float *brow = B + (size_t)k * N;   // contiguous in j
                        float *crow = C + (size_t)i * N;         // contiguous in j
                        for (int j = jc; j < jmax; j++) {
                            crow[j] += a * brow[j];              // stride-1, vectorizable
                        }
                    }
                }
            }
        }
    }
}
