#include "gemm.h"
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

// Block sizes (tune for L1/L2). KC*NR and MR*KC panels should stay cache-hot.
#ifndef MC
#define MC 128
#endif
#ifndef KC
#define KC 256
#endif
#ifndef NC
#define NC 512
#endif

#define MR 8   // microkernel rows
#define NR 8   // microkernel cols (2 NEON vectors of 4 floats)

// 8x8 register microkernel: C[i..i+7][j..j+7] += A[..][k0..k1) * B[k0..k1)[..]
// The 8x8 output tile lives entirely in 16 NEON registers across the whole k
// sweep; A and B are the only things streamed from cache. Each B vector load is
// reused by all 8 rows; each A scalar is reused by all 8 columns -> high
// arithmetic intensity, FMA-bound.
static void micro_8x8(const float *A, const float *B, float *C,
                      int N, int K, int i, int j, int k0, int k1) {
    float32x4_t c0[MR], c1[MR];
    for (int r = 0; r < MR; r++) {
        c0[r] = vld1q_f32(C + (size_t)(i + r) * N + j);
        c1[r] = vld1q_f32(C + (size_t)(i + r) * N + j + 4);
    }
    for (int k = k0; k < k1; k++) {
        const float32x4_t b0 = vld1q_f32(B + (size_t)k * N + j);
        const float32x4_t b1 = vld1q_f32(B + (size_t)k * N + j + 4);
        for (int r = 0; r < MR; r++) {
            const float32x4_t a = vdupq_n_f32(A[(size_t)(i + r) * K + k]);
            c0[r] = vfmaq_f32(c0[r], b0, a);
            c1[r] = vfmaq_f32(c1[r], b1, a);
        }
    }
    for (int r = 0; r < MR; r++) {
        vst1q_f32(C + (size_t)(i + r) * N + j, c0[r]);
        vst1q_f32(C + (size_t)(i + r) * N + j + 4, c1[r]);
    }
}

// Scalar accumulate for edge tiles (M or N not a multiple of 8).
static void edge(const float *A, const float *B, float *C,
                 int N, int K, int i0, int i1, int j0, int j1, int k0, int k1) {
    for (int i = i0; i < i1; i++)
        for (int k = k0; k < k1; k++) {
            float a = A[(size_t)i * K + k];
            const float *brow = B + (size_t)k * N;
            float *crow = C + (size_t)i * N;
            for (int j = j0; j < j1; j++)
                crow[j] += a * brow[j];
        }
}

void gemm_neon(const float *A, const float *B, float *C, int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int jc = 0; jc < N; jc += NC) {
        int jmax = jc + NC < N ? jc + NC : N;
        for (int kc = 0; kc < K; kc += KC) {
            int kmax = kc + KC < K ? kc + KC : K;
            for (int ic = 0; ic < M; ic += MC) {
                int imax = ic + MC < M ? ic + MC : M;
                // walk the block in MR x NR microkernel tiles
                for (int i = ic; i < imax; i += MR) {
                    int mr = imax - i < MR ? imax - i : MR;
                    for (int j = jc; j < jmax; j += NR) {
                        int nr = jmax - j < NR ? jmax - j : NR;
                        if (mr == MR && nr == NR)
                            micro_8x8(A, B, C, N, K, i, j, kc, kmax);
                        else
                            edge(A, B, C, N, K, i, i + mr, j, j + nr, kc, kmax);
                    }
                }
            }
        }
    }
}

#else  // No NEON: fall back to the blocked scalar version.
void gemm_neon(const float *A, const float *B, float *C, int M, int N, int K) {
    gemm_blocked(A, B, C, M, N, K);
}
#endif
