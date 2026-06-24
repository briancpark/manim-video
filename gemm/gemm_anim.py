"""Manim visualization of the GEMM optimizations (matches gemm_*.c).

  Tiling      -> cache blocking: naive streams all of B from DRAM over and over;
                 the blocked version keeps a tile resident in L1 and reuses it.
  Microkernel -> the NEON 8x8 register tile: a rank-1 (outer-product) update
                 per k, accumulated entirely in registers.
  Roofline    -> why each step helps: raising arithmetic intensity walks you
                 right toward the compute ceiling.

Render (Manim Community):
  .../manim-ce/bin/manim -qh --format mp4 gemm_anim.py Tiling
  .../manim-ce/bin/manim -qh --format mp4 gemm_anim.py Microkernel
  .../manim-ce/bin/manim -qh --format mp4 gemm_anim.py Roofline
"""
from manim import *

A_C, B_C, C_C, HL = BLUE_B, GREEN_B, YELLOW_E, ORANGE


def grid(rows, cols, cell=0.4, color=GREY_B, fill=BLACK, fop=0.0):
    g = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(cell, stroke_width=1.1, stroke_color=color,
                        fill_color=fill, fill_opacity=fop)
            sq.move_to([c * cell, -r * cell, 0])
            g.add(sq)
    g.rows, g.cols, g.cell = rows, cols, cell
    g.cellat = lambda r, c: g[r * cols + c]
    g.move_to(ORIGIN)
    return g


def mono(s, scale=0.42, color=WHITE, weight=NORMAL):
    return Text(s, font="Menlo", color=color, weight=weight).scale(scale)


# ===========================================================================
class Tiling(Scene):
    """Naive re-streams all of B from DRAM; blocking keeps a tile hot in L1."""

    def construct(self):
        title = mono("cache blocking: keep the working set in L1", 0.5, WHITE, BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        n = 8
        C = grid(n, n, 0.34, color=C_C).shift(LEFT * 4.2 + DOWN * 0.4)
        B = grid(n, n, 0.34, color=B_C).shift(RIGHT * 3.4 + DOWN * 0.4)
        cl = mono("C (output)", 0.34, C_C).next_to(C, UP, buff=0.2)
        bl = mono("B (in DRAM)", 0.34, B_C).next_to(B, UP, buff=0.2)
        self.play(FadeIn(C), FadeIn(B), Write(cl), Write(bl))

        # --- naive: each row of C sweeps ALL of B; DRAM traffic explodes ---
        sub = mono("naive: every row of C reads all of B  ->  B streamed M times",
                   0.36, RED).to_edge(DOWN, buff=0.9)
        self.play(FadeIn(sub))
        ctr = mono("DRAM passes over B: 0", 0.36, RED).next_to(title, DOWN, buff=0.3)
        self.play(Write(ctr))
        for r in range(3):
            row = VGroup(*[C.cellat(r, c) for c in range(n)])
            self.play(row.animate.set_fill(C_C, 0.5),
                      B.animate.set_fill(B_C, 0.55), run_time=0.4)
            self.play(B.animate.set_fill(B_C, 0.0), run_time=0.25)
            self.play(Transform(ctr, mono(f"DRAM passes over B: {r+1}", 0.36, RED)
                                .move_to(ctr)), run_time=0.2)
        dots = mono("... x M rows", 0.36, RED).next_to(C, DOWN, buff=0.25)
        self.play(FadeIn(dots))
        self.wait(0.6)
        self.play(FadeOut(VGroup(sub, dots)),
                  C.animate.set_fill(C_C, 0.0), B.animate.set_fill(B_C, 0.0))

        # --- blocked: one block of B sits in an L1 box, reused by a C panel ---
        sub2 = mono("blocked: load a B tile into L1 once, reuse it for many C",
                    0.36, GREEN_B).to_edge(DOWN, buff=0.9)
        self.play(FadeIn(sub2),
                  Transform(bl, mono("B (tiled)", 0.34, B_C).move_to(bl)))
        bblk = VGroup(*[B.cellat(r, c) for r in range(2, 6) for c in range(2, 6)])
        box = SurroundingRectangle(bblk, color=HL, stroke_width=5, buff=0.0)
        l1 = mono("L1 cache", 0.32, HL).next_to(box, UP, buff=0.12)
        self.play(bblk.animate.set_fill(HL, 0.45), Create(box), Write(l1))
        self.play(Transform(ctr, mono("DRAM passes over B: 1  (then reuse from L1)",
                                      0.36, GREEN_B).move_to(ctr)))

        cpanel = VGroup(*[C.cellat(r, c) for r in range(2, 6) for c in range(n)])
        reuse = mono("reused", 0.3, HL)
        for _ in range(4):
            self.play(LaggedStart(*[c.animate.set_fill(C_C, 0.6) for c in cpanel],
                                  lag_ratio=0.02, run_time=0.5),
                      Indicate(box, color=HL, scale_factor=1.05))
            self.play(cpanel.animate.set_fill(C_C, 0.2), run_time=0.2)
        self.wait(1.0)


# ===========================================================================
class Microkernel(Scene):
    """The NEON 8x8 register tile: one rank-1 outer-product update per k."""

    def construct(self):
        title = mono("NEON microkernel: an 8x8 tile lives in registers", 0.46, WHITE, BOLD)
        title.to_edge(UP, buff=0.4)
        self.play(Write(title))

        # A column (8x1), B row (1x8), C tile (8x8)
        Acol = grid(8, 1, 0.4, color=A_C).shift(LEFT * 4.3 + DOWN * 0.3)
        Ctile = grid(8, 8, 0.4, color=C_C).shift(RIGHT * 1.0 + DOWN * 0.3)
        Brow = grid(1, 8, 0.4, color=B_C).next_to(Ctile, UP, buff=0.5).align_to(Ctile, LEFT)
        al = mono("A[:,k]", 0.32, A_C).next_to(Acol, UP, buff=0.2)
        bl = mono("B[k,:]", 0.32, B_C).next_to(Brow, UP, buff=0.15)
        cl = mono("C  (8x8 accumulators)", 0.32, C_C).next_to(Ctile, DOWN, buff=0.25)
        self.play(FadeIn(Acol), FadeIn(Brow), FadeIn(Ctile), Write(al), Write(bl), Write(cl))

        eq = mono("for each k:   C[i,j] += A[i,k] * B[k,j]   (rank-1 update)",
                  0.38, HL).to_edge(DOWN, buff=0.5)
        self.play(Write(eq))

        # show 3 successive rank-1 updates (k = 0,1,2)
        for k in range(3):
            self.play(Acol.animate.set_fill(A_C, 0.5), Brow.animate.set_fill(B_C, 0.5),
                      run_time=0.3)
            # outer product: every C[i,j] gets A[i]*B[j]
            anims = []
            for i in range(8):
                for j in range(8):
                    anims.append(Ctile.cellat(i, j).animate.set_fill(C_C, 0.15 + 0.25 * (k + 1)))
            self.play(LaggedStart(*anims, lag_ratio=0.004, run_time=0.8))
            self.play(Acol.animate.set_fill(A_C, 0.0), Brow.animate.set_fill(B_C, 0.0),
                      run_time=0.2)
        note = mono("8 broadcasts x 2 vector-FMAs per k  ->  FMA-bound, no C reload",
                    0.34, GREY_A).next_to(eq, UP, buff=0.3)
        self.play(Ctile.animate.set_fill(C_C, 0.9), FadeIn(note))
        self.wait(1.0)


# ===========================================================================
class Roofline(Scene):
    """Roofline: raising arithmetic intensity walks you toward the compute roof."""

    def construct(self):
        title = mono("roofline: intensity vs. throughput", 0.5, WHITE, BOLD)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title))

        # log-log axes (we map log10 by hand onto linear axes)
        ax = Axes(x_range=[-1, 3, 1], y_range=[0, 4, 1], x_length=9, y_length=5.2,
                  axis_config={"include_numbers": False, "stroke_width": 2}).shift(DOWN * 0.4)
        xl = mono("arithmetic intensity  (FLOP/byte, log)", 0.32).next_to(ax, DOWN, buff=0.3)
        yl = mono("GFLOP/s (log)", 0.32).rotate(PI / 2).next_to(ax, LEFT, buff=0.2)
        self.play(Create(ax), Write(xl), Write(yl))

        BW = 200.0                      # GB/s  (memory roof slope in log space: +1)
        mem = ax.plot(lambda x: min(x + 2.3, 2.0), x_range=[-1, 3], color=BLUE_C,
                      stroke_width=3)
        comp = DashedLine(ax.c2p(-1, 2.0), ax.c2p(3, 2.0), color=GREEN_C, stroke_width=3)
        mem_l = mono("memory-bound (BW)", 0.3, BLUE_C).next_to(ax.c2p(-0.3, 1.9), UL, buff=0.05)
        comp_l = mono("NEON compute roof", 0.3, GREEN_C).next_to(ax.c2p(2.2, 2.0), UP, buff=0.1)
        self.play(Create(mem), Create(comp), Write(mem_l), Write(comp_l))

        # points: (log10 AI, log10 GFLOP/s, label, color)
        pts = [(-0.7, 0.2, "naive 1.6", RED),
               (0.2, 1.3, "blocked 20", YELLOW_E),
               (0.9, 1.8, "neon 64", ORANGE),
               (2.6, 3.3, "AMX BLAS 2000", PURPLE)]
        dots, labels = VGroup(), VGroup()
        for x, y, t, c in pts:
            d = Dot(ax.c2p(x, y), color=c, radius=0.09)
            lab = mono(t, 0.28, c).next_to(d, RIGHT, buff=0.12)
            dots.add(d); labels.add(lab)
        for i in range(len(pts)):
            self.play(GrowFromCenter(dots[i]), FadeIn(labels[i]), run_time=0.5)

        arrow = Arrow(ax.c2p(-0.7, 0.2), ax.c2p(0.9, 1.8), buff=0.15, color=WHITE,
                      stroke_width=3)
        note = mono("blocking raises intensity\n-> climb toward the roof", 0.32, GREY_A) \
            .move_to(ax.c2p(0.1, 0.55))
        self.play(GrowArrow(arrow), FadeIn(note))
        self.wait(1.2)
