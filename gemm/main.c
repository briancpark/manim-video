#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "gemm.h"

#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static float *alloc_fill(size_t n, unsigned *seed) {
    float *p = malloc(n * sizeof(float));
    for (size_t i = 0; i < n; i++)
        p[i] = (float)((int)(rand_r(seed) % 2001) - 1000) / 1000.0f;
    return p;
}

static double max_rel_diff(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        double s = fabs((double)a[i]) + 1e-6;
        if (d / s > m) m = d / s;
    }
    return m;
}

typedef void (*gemm_fn)(const float *, const float *, float *, int, int, int);

static double bench(gemm_fn fn, const float *A, const float *B, float *C,
                    int M, int N, int K, int iters) {
    fn(A, B, C, M, N, K);                       // warm up
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) fn(A, B, C, M, N, K);
    return (now_sec() - t0) / iters;
}

int main(int argc, char **argv) {
    int M = 1024, N = 1024, K = 1024;
    if (argc == 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }

    unsigned seed = 7;
    float *A = alloc_fill((size_t)M * K, &seed);
    float *B = alloc_fill((size_t)K * N, &seed);
    float *C0 = malloc((size_t)M * N * sizeof(float));   // reference
    float *C = malloc((size_t)M * N * sizeof(float));

    double gflop = 2.0 * M * N * K * 1e-9;
    int iters = (M * N * K > 200000000) ? 5 : 20;

    printf("sgemm  C[%dx%d] = A[%dx%d] * B[%dx%d]   (%.2f GFLOP/call)\n",
           M, N, M, K, K, N, gflop);

    gemm_naive(A, B, C0, M, N, K);   // ground truth

    printf("\n%-12s %10s %10s %9s %8s\n", "impl", "ms", "GFLOP/s", "speedup", "max_rel");
    double t_naive = bench(gemm_naive, A, B, C, M, N, K, iters < 5 ? iters : 3);
    printf("%-12s %10.2f %10.1f %8s %8s\n", "naive", t_naive * 1e3,
           gflop / t_naive, "1.00x", "-");

    double t_blk = bench(gemm_blocked, A, B, C, M, N, K, iters);
    printf("%-12s %10.2f %10.1f %7.2fx %8.1e\n", "blocked", t_blk * 1e3,
           gflop / t_blk, t_naive / t_blk, max_rel_diff(C0, C, (size_t)M * N));

    double t_neon = bench(gemm_neon, A, B, C, M, N, K, iters);
    printf("%-12s %10.2f %10.1f %7.2fx %8.1e\n", "neon", t_neon * 1e3,
           gflop / t_neon, t_naive / t_neon, max_rel_diff(C0, C, (size_t)M * N));

#ifdef USE_ACCELERATE
    double t_bl;
    {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                    1.0f, A, K, B, N, 0.0f, C, N);                 // warm up
        double t0 = now_sec();
        for (int i = 0; i < iters; i++)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                        1.0f, A, K, B, N, 0.0f, C, N);
        t_bl = (now_sec() - t0) / iters;
    }
    printf("%-12s %10.2f %10.1f %7.2fx %8.1e   <- Apple Accelerate (AMX)\n", "BLAS",
           t_bl * 1e3, gflop / t_bl, t_naive / t_bl, max_rel_diff(C0, C, (size_t)M * N));
    printf("\nnote: Accelerate runs on the AMX matrix coprocessor (dedicated HW),\n"
           "      not NEON. Our NEON kernel is single-core SIMD -- a different league.\n");
#endif

    free(A); free(B); free(C0); free(C);
    return 0;
}
