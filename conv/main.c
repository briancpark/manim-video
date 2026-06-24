#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "conv2d.h"

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

static double max_abs_diff(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double dd = fabs((double)a[i] - (double)b[i]);
        if (dd > m) m = dd;
    }
    return m;
}

// 2 * Cout * OH * OW * Cin * KH * KW flops (mul + add per MAC).
static double conv_gflop(const conv_dims *d) {
    double macs = (double)d->Cout * conv_OH(d) * conv_OW(d) *
                  d->Cin * d->KH * d->KW;
    return 2.0 * macs * 1e-9;
}

static double bench(void (*fn)(const float *, const float *, const float *,
                              float *, const conv_dims *),
                    const float *in, const float *w, const float *b,
                    float *out, const conv_dims *d, int iters) {
    fn(in, w, b, out, d);            // warm up
    double t0 = now_sec();
    for (int i = 0; i < iters; i++) fn(in, w, b, out, d);
    return (now_sec() - t0) / iters;
}

// One bench row, used by both the pretty printer and the CSV sweep.
static void run_one(conv_dims d, int iters, int csv) {
    const int OH = conv_OH(&d), OW = conv_OW(&d);
    if (OH <= 0 || OW <= 0) return;

    unsigned seed = 1234;
    size_t in_n  = (size_t)d.Cin * d.H * d.W;
    size_t w_n   = (size_t)d.Cout * d.Cin * d.KH * d.KW;
    size_t out_n = (size_t)d.Cout * OH * OW;

    float *in   = alloc_fill(in_n, &seed);
    float *w    = alloc_fill(w_n, &seed);
    float *bias = alloc_fill(d.Cout, &seed);
    float *o_ref = malloc(out_n * sizeof(float));
    float *o_nn  = malloc(out_n * sizeof(float));

    double t_ref = bench(conv2d_naive, in, w, bias, o_ref, &d, iters);
    double t_nn  = bench(conv2d_neon,  in, w, bias, o_nn,  &d, iters);
    double gf = conv_gflop(&d);

    if (csv) {
        // label,Cin,Cout,H,W,K,gflops_naive,gflops_neon,speedup
        printf("%dc_%dk,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f\n",
               d.Cin, d.KH, d.Cin, d.Cout, d.H, d.W, d.KH,
               gf / t_ref, gf / t_nn, t_ref / t_nn);
    } else {
        printf("conv2d  Cin=%d Cout=%d  in=%dx%d  k=%dx%d  out=%dx%d\n",
               d.Cin, d.Cout, d.H, d.W, d.KH, d.KW, OH, OW);
        double diff = max_abs_diff(o_ref, o_nn, out_n);
        printf("\n%-8s %10s %10s %8s\n", "impl", "ms/iter", "GFLOP/s", "speedup");
        printf("%-8s %10.3f %10.2f %8s\n", "naive", t_ref * 1e3, gf / t_ref, "1.00x");
        printf("%-8s %10.3f %10.2f %7.2fx\n", "neon", t_nn * 1e3, gf / t_nn, t_ref / t_nn);
        printf("\nmax|naive-neon| = %.3e  (%s)\n", diff, diff < 1e-3 ? "PASS" : "FAIL");
    }
    free(in); free(w); free(bias); free(o_ref); free(o_nn);
}

int main(int argc, char **argv) {
    // Sweep mode: emit CSV across a range of channel counts for plotting.
    if (argc == 2 && strcmp(argv[1], "--sweep") == 0) {
        printf("label,Cin,Cout,H,W,K,gflops_naive,gflops_neon,speedup\n");
        int chans[] = {8, 16, 32, 64, 128};
        int ks[] = {1, 3, 5};
        for (size_t k = 0; k < sizeof(ks) / sizeof(ks[0]); k++)
            for (size_t c = 0; c < sizeof(chans) / sizeof(chans[0]); c++) {
                int ch = chans[c], kk = ks[k];
                conv_dims d = {ch, ch, 128 + kk - 1, 128 + kk - 1, kk, kk};
                run_one(d, 10, 1);
            }
        return 0;
    }

    conv_dims d = {.Cin = 64, .Cout = 64, .H = 130, .W = 130, .KH = 3, .KW = 3};
    if (argc == 7) {
        d.Cin = atoi(argv[1]); d.Cout = atoi(argv[2]);
        d.H = atoi(argv[3]); d.W = atoi(argv[4]);
        d.KH = atoi(argv[5]); d.KW = atoi(argv[6]);
    }
    if (conv_OH(&d) <= 0 || conv_OW(&d) <= 0) {
        fprintf(stderr, "kernel larger than input\n"); return 1;
    }
    run_one(d, 20, 0);
    return 0;
}
