"""Manim visualization of the im2col -> GEMM lowering of conv2d.

The trick every fast conv (cuDNN, oneDNN, BLAS-backed) uses: turn convolution
into a single matrix multiply.

  Im2Col      -> each sliding KH*KW*Cin patch is FLATTENED into one COLUMN of a
                 big matrix X of shape [Cin*KH*KW] x [OH*OW].
  Im2ColGEMM  -> filters become rows of W [Cout] x [Cin*KH*KW]; then
                 Y = W @ X  is the whole convolution, as one GEMM.

Render:
  .../manim-ce/bin/manim -ql --format mp4 conv_im2col.py Im2Col
  .../manim-ce/bin/manim -ql --format mp4 conv_im2col.py Im2ColGEMM
"""
from manim import *

IN_C, OUT_C, W_C = BLUE_B, GREEN_B, YELLOW_E


def grid(rows, cols, cell=0.5, color=GREY_B, fill=BLACK, fop=0.0):
    g = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(cell, stroke_width=1.3, stroke_color=color,
                        fill_color=fill, fill_opacity=fop)
            sq.move_to([c * cell, -r * cell, 0])
            g.add(sq)
    g.rows, g.cols, g.cell = rows, cols, cell
    g.cellat = lambda r, c: g[r * cols + c]
    g.move_to(ORIGIN)
    return g


def mtext(s, scale=0.42, color=WHITE):
    return Text(s, font="Menlo", color=color).scale(scale)


# 9 colors so each position inside a 3x3 patch is identifiable as it flattens.
def patch_palette(n):
    return color_gradient([BLUE_D, TEAL, GREEN, YELLOW, GOLD, RED], n)


# ===========================================================================
class Im2Col(Scene):
    """Each sliding patch is flattened into one column of the im2col matrix."""

    def construct(self):
        H = W = 5
        K = 3
        OH = OW = H - K + 1     # 3
        P = OH * OW             # 9 patches  (columns)
        L = K * K              # 9 per patch (rows)
        PAL = patch_palette(L)

        title = mtext("im2col : flatten each KxK patch into a column", 0.5)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title))

        inp = grid(H, W, 0.52, color=IN_C).to_edge(LEFT, buff=0.7).shift(DOWN * 0.3)
        inlab = mtext("input  5x5", 0.4, IN_C).next_to(inp, UP, buff=0.2)

        col = grid(L, P, 0.34, color=GREY_B).to_edge(RIGHT, buff=0.7).shift(DOWN * 0.3)
        collab = mtext("im2col  (K*K) x (OH*OW) = 9 x 9", 0.36, WHITE) \
            .next_to(col, UP, buff=0.2)
        self.play(FadeIn(inp), Write(inlab), FadeIn(col), Write(collab))

        win = SurroundingRectangle(
            VGroup(inp.cellat(0, 0), inp.cellat(K - 1, K - 1)),
            color=ORANGE, stroke_width=5, buff=0.0)
        self.add(win)

        def lower_patch(p, detailed):
            pr, pc = p // OW, p % OW                 # patch top-left in output coords
            self.play(win.animate.move_to(
                VGroup(inp.cellat(pr, pc), inp.cellat(pr + K - 1, pc + K - 1))
                .get_center()), run_time=0.4 if detailed else 0.18)

            copies = VGroup()
            for ky in range(K):
                for kx in range(K):
                    idx = ky * K + kx                # row in the column (flatten order)
                    src = inp.cellat(pr + ky, pc + kx)
                    cp = src.copy().set_fill(PAL[idx], 0.95).set_stroke(PAL[idx], 1.5)
                    copies.add(cp)
            self.add(copies)

            anims = []
            for idx in range(L):
                dst = col.cellat(idx, p)
                anims.append(copies[idx].animate.move_to(dst.get_center())
                             .scale(col.cell / inp.cell))
            if detailed:
                self.play(LaggedStart(*anims, lag_ratio=0.12, run_time=1.3))
            else:
                self.play(*anims, run_time=0.35)
            # bake color into the destination matrix, drop the moving copies
            for idx in range(L):
                col.cellat(idx, p).set_fill(PAL[idx], 0.95)
            self.remove(copies)

        lower_patch(0, True)
        lower_patch(1, True)
        for p in range(2, P):
            lower_patch(p, False)

        note = mtext("each column = one output pixel's receptive field", 0.38, GREY_A)
        note.next_to(col, DOWN, buff=0.35)
        self.play(FadeOut(win), FadeIn(note))
        self.wait(1.3)


# ===========================================================================
class Im2ColGEMM(Scene):
    """Y = W @ X : convolution as one matrix multiply, then reshape back."""

    def construct(self):
        Cout = 3
        L = 9          # Cin*KH*KW
        P = 9          # OH*OW
        OH = OW = 3

        title = mtext("im2col + GEMM :   Y = W @ X", 0.5).to_edge(UP, buff=0.3)
        self.play(Write(title))

        PAL = patch_palette(L)

        # W : [Cout x L]   filters as rows
        Wm = grid(Cout, L, 0.34, color=W_C).shift(LEFT * 3.2 + UP * 1.9)
        Wl = mtext("W  [Cout x L]", 0.34, W_C).next_to(Wm, UP, buff=0.15)
        for r in range(Cout):
            for c in range(L):
                Wm.cellat(r, c).set_fill(W_C, 0.18 + 0.06 * r)

        # X : [L x P]   the im2col matrix (columns colored as in scene 1)
        Xm = grid(L, P, 0.34, color=GREY_B).shift(RIGHT * 2.4 + DOWN * 1.3)
        Xl = mtext("X  [L x P]  (im2col)", 0.34).next_to(Xm, UP, buff=0.15)
        for r in range(L):
            for c in range(P):
                Xm.cellat(r, c).set_fill(PAL[r], 0.85)

        # Y : [Cout x P]   result, aligned under W and right of X
        Ym = grid(Cout, P, 0.34, color=OUT_C).shift(RIGHT * 2.4 + UP * 1.9)
        Yl = mtext("Y  [Cout x P]", 0.34, OUT_C).next_to(Ym, UP, buff=0.15)

        self.play(FadeIn(Wm), Write(Wl), FadeIn(Xm), Write(Xl), FadeIn(Ym), Write(Yl))

        eq = mtext("Y[i,j] = sum_k  W[i,k] * X[k,j]", 0.4, HL := ORANGE)
        eq.to_edge(DOWN, buff=0.4)
        self.play(Write(eq))

        # Animate one output cell as the dot product of a W row and an X column.
        for (i, j) in [(0, 0), (1, 4), (2, 8)]:
            wrow = VGroup(*[Wm.cellat(i, k) for k in range(L)])
            xcol = VGroup(*[Xm.cellat(k, j) for k in range(L)])
            wbox = SurroundingRectangle(wrow, color=W_C, stroke_width=4, buff=0.02)
            xbox = SurroundingRectangle(xcol, color=BLUE, stroke_width=4, buff=0.02)
            ycell = Ym.cellat(i, j)
            self.play(Create(wbox), Create(xbox))
            self.play(LaggedStart(
                *[Indicate(Wm.cellat(i, k), color=RED, scale_factor=1.2) for k in range(L)],
                *[Indicate(Xm.cellat(k, j), color=RED, scale_factor=1.2) for k in range(L)],
                lag_ratio=0.06, run_time=1.0),
                ycell.animate.set_fill(OUT_C, 0.95))
            self.play(FadeOut(wbox), FadeOut(xbox), run_time=0.3)

        # fill the rest of Y instantly
        self.play(*[Ym.cellat(r, c).animate.set_fill(OUT_C, 0.95)
                    for r in range(Cout) for c in range(P)], run_time=0.5)

        # Reshape: one output channel's P values fold back into an OH x OW image.
        self.play(eq.animate.become(
            mtext("reshape: each Y row (P values) -> OH x OW feature map", 0.38, GREY_A)
            .to_edge(DOWN, buff=0.4)))
        spatial = grid(OH, OW, 0.5, color=OUT_C).shift(LEFT * 3.4 + DOWN * 1.4)
        sl = mtext("Y[0]  ->  3x3 map", 0.34, OUT_C).next_to(spatial, UP, buff=0.15)
        self.play(FadeIn(spatial), Write(sl))
        moves = []
        for j in range(P):
            r, c = j // OW, j % OW
            cp = Ym.cellat(0, j).copy().set_fill(OUT_C, 0.9)
            self.add(cp)
            moves.append(cp.animate.move_to(spatial.cellat(r, c).get_center())
                         .scale(spatial.cell / Ym.cell))
        self.play(LaggedStart(*moves, lag_ratio=0.08, run_time=1.6))
        self.wait(1.3)
