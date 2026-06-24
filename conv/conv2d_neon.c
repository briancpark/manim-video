#include "conv2d.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

/*============================================================================
  conv2d_neon — direct multi-channel convolution, vectorized over OUTPUT WIDTH
============================================================================*/
/*
  THE PROBLEM
  -----------
  For every output pixel we sum Cin*KH*KW multiply-accumulates (MACs):

      out[oc][oy][ox] = bias[oc]
                      + Σ_ic Σ_ky Σ_kx  in[ic][oy+ky][ox+kx] * w[oc][ic][ky][kx]

  The naive code walks one output pixel to completion (innermost loops over
  ic,ky,kx) — every weight is re-loaded for every pixel and nothing is reused
  in registers.

  THE KEY IDEA: vectorize across OX (output width)
  ------------------------------------------------
  Neighboring output pixels along a row share the SAME weight and read
  ADJACENT input pixels. So we flip the loop nest: fix one weight w[ic][ky][kx],
  broadcast it into all 4 lanes of a NEON register, and apply it to 4 (or 16)
  consecutive output pixels at once with a single fused-multiply-add.

      weight scalar  w ──────────────┐ (broadcast to every lane)
                                      ▼
      input  row   [ x0  x1  x2  x3  x4  x5 ... ]   contiguous, stride-1
                     └──┬──┴──┬──┴──┬──┴──┬─┘
                     vld1q_f32 loads 4 lanes at a time
                                      │  vfmaq:  acc += x * w   (per lane)
                                      ▼
      output lanes  [ o0  o1  o2  o3 ]   ox .. ox+3  accumulate in registers

  ONE accumulator vector across the (ic,ky,kx) reduction:

        acc(4 lanes)
          ^   for each ic:            <- input channel
          |     for each ky:          <- kernel row  (picks input row oy+ky)
          |       for each kx:        <- kernel col  (shifts the load by kx)
          |         w = splat(weight[ic][ky][kx])
          +──────── acc = fma(acc, load(input_row + kx), w)   // 4 MACs/instr

  WIDTH BLOCKING (the x16 main loop)
  ----------------------------------
  A single FMA has ~4-cycle latency but the core can issue several per cycle.
  One accumulator chain would stall waiting on its own previous FMA. So we keep
  FOUR independent accumulators (a0..a3 = 16 output pixels) in flight; while a0
  is still computing, a1/a2/a3 issue — latency is hidden, throughput is FMA-bound.

      ox:   [ a0: 0..3 ][ a1: 4..7 ][ a2: 8..11 ][ a3: 12..15 ]   per iteration
            \________________ 4 parallel FMA chains _____________/
*/

void conv2d_neon(const float *input, const float *weight, const float *bias,
                 float *output, const conv_dims *d) {
    const int OH = conv_OH(d), OW = conv_OW(d);
    const int H = d->H, W = d->W, KH = d->KH, KW = d->KW;
    const int Cin = d->Cin, Cout = d->Cout;

    /* LOOP 1 — output channels. Each oc uses its own filter bank w[oc][*] and
       writes an independent OH×OW output plane. Fully parallel (good thread axis). */
    for (int oc = 0; oc < Cout; oc++) {
        const float b = bias ? bias[oc] : 0.0f;
        const float32x4_t vb = vdupq_n_f32(b);   /* bias splat -> seeds accumulators */

        /* LOOP 2 — output rows. Fixing oy fixes which input rows (oy+ky) we touch. */
        for (int oy = 0; oy < OH; oy++) {
            float *out = output + (((size_t)oc * OH) + oy) * OW;

            int ox = 0;

            /* LOOP 3a — WIDTH, 16 pixels per step (4 NEON vectors).
               a0..a3 hold 16 output pixels live in registers for the whole
               (ic,ky,kx) reduction below; they are only stored once, at the end. */
            for (; ox + 16 <= OW; ox += 16) {
                float32x4_t a0 = vb, a1 = vb, a2 = vb, a3 = vb;  /* start from bias */

                /* LOOP 4 — input channels: accumulate each ic's contribution. */
                for (int ic = 0; ic < Cin; ic++) {
                    const float *in = input + (size_t)ic * H * W;
                    const float *wt = weight + (((size_t)oc * Cin + ic) * KH) * KW;

                    /* LOOP 5 — kernel ROWS: ky selects input row (oy+ky). */
                    for (int ky = 0; ky < KH; ky++) {
                        const float *row  = in + (size_t)(oy + ky) * W + ox; /* window start */
                        const float *wrow = wt + ky * KW;

                        /* LOOP 6 — kernel COLS: kx shifts the input window by kx.
                           Broadcast one weight, FMA it into all 4 accumulators.
                           The 4 loads overlap (p+0,4,8,12) — same weight, adjacent data. */
                        for (int kx = 0; kx < KW; kx++) {
                            const float32x4_t vw = vdupq_n_f32(wrow[kx]); /* splat weight */
                            const float *p = row + kx;
                            a0 = vfmaq_f32(a0, vld1q_f32(p + 0),  vw);   /* out[ox  .. ox+3 ] */
                            a1 = vfmaq_f32(a1, vld1q_f32(p + 4),  vw);   /* out[ox+4.. ox+7 ] */
                            a2 = vfmaq_f32(a2, vld1q_f32(p + 8),  vw);   /* out[ox+8.. ox+11] */
                            a3 = vfmaq_f32(a3, vld1q_f32(p + 12), vw);   /* out[ox+12..ox+15] */
                        }
                    }
                }
                /* reduction done for these 16 pixels -> commit to memory */
                vst1q_f32(out + ox + 0,  a0);
                vst1q_f32(out + ox + 4,  a1);
                vst1q_f32(out + ox + 8,  a2);
                vst1q_f32(out + ox + 12, a3);
            }

            /* LOOP 3b — WIDTH tail, 4 pixels per step (same idea, one accumulator). */
            for (; ox + 4 <= OW; ox += 4) {
                float32x4_t a0 = vb;
                for (int ic = 0; ic < Cin; ic++) {
                    const float *in = input + (size_t)ic * H * W;
                    const float *wt = weight + (((size_t)oc * Cin + ic) * KH) * KW;
                    for (int ky = 0; ky < KH; ky++) {
                        const float *row  = in + (size_t)(oy + ky) * W + ox;
                        const float *wrow = wt + ky * KW;
                        for (int kx = 0; kx < KW; kx++)
                            a0 = vfmaq_f32(a0, vld1q_f32(row + kx), vdupq_n_f32(wrow[kx]));
                    }
                }
                vst1q_f32(out + ox, a0);
            }

            /* LOOP 3c — WIDTH remainder (< 4 pixels): plain scalar MACs. */
            for (; ox < OW; ox++) {
                float acc = b;
                for (int ic = 0; ic < Cin; ic++) {
                    const float *in = input + (size_t)ic * H * W;
                    const float *wt = weight + (((size_t)oc * Cin + ic) * KH) * KW;
                    for (int ky = 0; ky < KH; ky++)
                        for (int kx = 0; kx < KW; kx++)
                            acc += in[(oy + ky) * W + (ox + kx)] * wt[ky * KW + kx];
                }
                out[ox] = acc;
            }
        }
    }
}

#else  /* No NEON (e.g. x86): fall back to the scalar reference. */
void conv2d_neon(const float *input, const float *weight, const float *bias,
                 float *output, const conv_dims *d) {
    conv2d_naive(input, weight, bias, output, d);
}
#endif
