// Thin CLI around conv2d_neon so Python can feed it real images.
//
// Binary protocol (little-endian, matches numpy tofile):
//   in  file: int32 [Cin,Cout,H,W,KH,KW]  then  Cin*H*W floats  then  weights
//   out file: Cout*OH*OW floats
//
// Usage: ./conv_filter <in.bin> <out.bin>

#include <stdio.h>
#include <stdlib.h>
#include "conv2d.h"

static void *xread(FILE *f, size_t n) {
    void *p = malloc(n);
    if (!p || fread(p, 1, n, f) != n) { fprintf(stderr, "read failed\n"); exit(1); }
    return p;
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <in.bin> <out.bin>\n", argv[0]); return 1; }

    FILE *fi = fopen(argv[1], "rb");
    if (!fi) { perror("open in"); return 1; }

    int hdr[6];
    if (fread(hdr, sizeof(int), 6, fi) != 6) { fprintf(stderr, "bad header\n"); return 1; }
    conv_dims d = {hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]};

    size_t in_n  = (size_t)d.Cin * d.H * d.W;
    size_t w_n   = (size_t)d.Cout * d.Cin * d.KH * d.KW;
    size_t out_n = (size_t)d.Cout * conv_OH(&d) * conv_OW(&d);

    float *in = xread(fi, in_n * sizeof(float));
    float *w  = xread(fi, w_n * sizeof(float));
    fclose(fi);

    float *out = malloc(out_n * sizeof(float));
    conv2d_neon(in, w, NULL, out, &d);

    FILE *fo = fopen(argv[2], "wb");
    if (!fo) { perror("open out"); return 1; }
    fwrite(out, sizeof(float), out_n, fo);
    fclose(fo);

    free(in); free(w); free(out);
    return 0;
}
