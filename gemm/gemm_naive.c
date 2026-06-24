#include "gemm.h"
#include <string.h>

// Textbook ijk order. The inner loop walks B down a column (B[k*N + j] strides
// by N each step), so it misses cache badly for large matrices. This is the
// slow baseline and the bit-exact-ish ground truth for the others.
void gemm_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[(size_t)i * K + k] * B[(size_t)k * N + j];
            }
            C[(size_t)i * N + j] = acc;
        }
    }
}
