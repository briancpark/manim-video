#include "conv2d.h"

void conv2d_naive(const float *input, const float *weight, const float *bias,
                  float *output, const conv_dims *d) {
    const int OH = conv_OH(d), OW = conv_OW(d);
    const int H = d->H, W = d->W, KH = d->KH, KW = d->KW, Cin = d->Cin;

    // Textbook 6-deep nest: compute one output pixel fully before moving on.
    // Simple and obviously correct — this is the bit-exact ground truth the
    // NEON path is checked against. It is slow because every weight is re-read
    // from memory for every output pixel (no register reuse, no SIMD).
    for (int oc = 0; oc < d->Cout; oc++) {            // output channel  (filter bank)
        for (int oy = 0; oy < OH; oy++) {             // output row
            for (int ox = 0; ox < OW; ox++) {         // output col  (one pixel per iter)
                float acc = bias ? bias[oc] : 0.0f;   // running sum for THIS pixel
                for (int ic = 0; ic < Cin; ic++) {    // reduce over input channels
                    const float *in = input + (size_t)ic * H * W;
                    const float *wt = weight + (((size_t)oc * Cin + ic) * KH) * KW;
                    for (int ky = 0; ky < KH; ky++) {        // kernel row
                        for (int kx = 0; kx < KW; kx++) {    // kernel col -> one MAC
                            acc += in[(oy + ky) * W + (ox + kx)] * wt[ky * KW + kx];
                        }
                    }
                }
                output[((size_t)oc * OH + oy) * OW + ox] = acc;   // store the pixel
            }
        }
    }
}
