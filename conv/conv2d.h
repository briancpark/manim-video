#ifndef CONV2D_H
#define CONV2D_H

#include <stddef.h>

// Layout (row-major, stride 1, "valid" convolution):
//   input  : [Cin][H][W]
//   weight : [Cout][Cin][KH][KW]
//   bias   : [Cout]            (may be NULL)
//   output : [Cout][OH][OW]  with OH = H-KH+1, OW = W-KW+1
//
// All tensors are 32-bit float, contiguous.

typedef struct {
    int Cin, Cout;
    int H, W;
    int KH, KW;
} conv_dims;

static inline int conv_OH(const conv_dims *d) { return d->H - d->KH + 1; }
static inline int conv_OW(const conv_dims *d) { return d->W - d->KW + 1; }

// Scalar reference implementation.
void conv2d_naive(const float *input, const float *weight, const float *bias,
                  float *output, const conv_dims *d);

// NEON-optimized implementation (falls back to naive off ARM).
void conv2d_neon(const float *input, const float *weight, const float *bias,
                 float *output, const conv_dims *d);

#endif // CONV2D_H
