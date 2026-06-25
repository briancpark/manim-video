"""Manim visualization of GEMM optimizations across CPU and NVIDIA Tensor Cores.

  Tiling      -> cache blocking: naive streams all of B from DRAM over and over;
                 the blocked version keeps a tile resident in L1 and reuses it.
  Microkernel -> the NEON 8x8 register tile: a rank-1 (outer-product) update
                 per k, accumulated entirely in registers.
  TensorCore  -> NVIDIA warp-level MMA: small matrix fragments are staged into
                 registers, then tensor cores perform D = A x B + C as a fused
                 matrix operation.
  Roofline    -> why each step helps: raising arithmetic intensity and using
                 specialized matrix hardware moves GEMM toward a much higher roof.

Render (Manim Community):
  manim -ql --format mp4 gemm_anim.py Tiling Microkernel TensorCore Roofline
"""
from manim import *

A_C, B_C, C_C, HL = BLUE_B, GREEN_B, YELLOW_E, ORANGE
TC_C, WARP_C, MEM_C = "#B67CFF", "#42E8D8", "#4D8DFF"


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


def fade_all(scene):
    if scene.mobjects:
        scene.play(FadeOut(Group(*scene.mobjects)), run_time=0.5)


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

        for k in range(3):
            self.play(Acol.animate.set_fill(A_C, 0.5), Brow.animate.set_fill(B_C, 0.5),
                      run_time=0.3)
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
class TensorCore(Scene):
    """NVIDIA tensor cores perform warp-level matrix multiply-accumulate."""

    def construct(self):
        title = mono("NVIDIA Tensor Cores: matrix multiply as one hardware operation",
                     0.43, WHITE, BOLD).to_edge(UP, buff=0.35)
        self.play(Write(title))

        # Global/shared memory staging on the left.
        global_box = RoundedRectangle(width=2.4, height=1.0, corner_radius=0.12,
                                      color=MEM_C, stroke_width=3).shift(LEFT * 4.7 + UP * 1.7)
        shared_box = RoundedRectangle(width=2.4, height=1.0, corner_radius=0.12,
                                      color=HL, stroke_width=3).next_to(global_box, DOWN, buff=0.75)
        reg_box = RoundedRectangle(width=2.4, height=1.0, corner_radius=0.12,
                                   color=WARP_C, stroke_width=3).next_to(shared_box, DOWN, buff=0.75)
        labels = VGroup(
            mono("HBM / L2", 0.32, MEM_C).move_to(global_box),
            mono("shared memory", 0.30, HL).move_to(shared_box),
            mono("warp fragments", 0.30, WARP_C).move_to(reg_box),
        )
        self.play(Create(global_box), Create(shared_box), Create(reg_box), FadeIn(labels))

        arrow1 = Arrow(global_box.get_bottom(), shared_box.get_top(), buff=0.08, color=MEM_C)
        arrow2 = Arrow(shared_box.get_bottom(), reg_box.get_top(), buff=0.08, color=HL)
        self.play(GrowArrow(arrow1), GrowArrow(arrow2))

        # Matrix fragments feeding a tensor core.
        Afrag = grid(4, 4, 0.25, color=A_C, fill=A_C, fop=0.18).shift(LEFT * 1.2 + UP * 1.35)
        Bfrag = grid(4, 4, 0.25, color=B_C, fill=B_C, fop=0.18).shift(RIGHT * 1.1 + UP * 1.35)
        Cfrag = grid(4, 4, 0.25, color=C_C, fill=C_C, fop=0.10).shift(RIGHT * 2.1 + DOWN * 1.1)
        tensor = RoundedRectangle(width=2.4, height=1.3, corner_radius=0.18,
                                  color=TC_C, fill_color=TC_C, fill_opacity=0.15,
                                  stroke_width=4).shift(RIGHT * 0.35 + DOWN * 0.05)
        tlabel = mono("tensor core\nMMA pipe", 0.32, TC_C, BOLD).move_to(tensor)
        self.play(FadeIn(Afrag), FadeIn(Bfrag), FadeIn(Cfrag), Create(tensor), FadeIn(tlabel))
        self.play(
            Write(mono("A fragment", 0.27, A_C).next_to(Afrag, UP, buff=0.12)),
            Write(mono("B fragment", 0.27, B_C).next_to(Bfrag, UP, buff=0.12)),
            Write(mono("C/D accum", 0.27, C_C).next_to(Cfrag, DOWN, buff=0.12)),
        )

        a_to_tc = Arrow(Afrag.get_right(), tensor.get_left(), buff=0.1, color=A_C)
        b_to_tc = Arrow(Bfrag.get_bottom(), tensor.get_top(), buff=0.1, color=B_C)
        tc_to_c = Arrow(tensor.get_right(), Cfrag.get_left(), buff=0.1, color=C_C)
        self.play(GrowArrow(a_to_tc), GrowArrow(b_to_tc), GrowArrow(tc_to_c))

        formula = mono("warp instruction:  D = A x B + C", 0.40, WHITE, BOLD).to_edge(DOWN, buff=0.75)
        detail = mono("many FMAs happen inside the tensor core each cycle; threads hold fragments, not scalar loops",
                      0.29, GREY_A).next_to(formula, UP, buff=0.25)
        self.play(Write(formula), FadeIn(detail))

        for alpha in (0.35, 0.65, 0.95):
            self.play(Afrag.animate.set_fill(A_C, alpha),
                      Bfrag.animate.set_fill(B_C, alpha),
                      Indicate(tensor, color=TC_C, scale_factor=1.05),
                      run_time=0.65)
            self.play(Cfrag.animate.set_fill(C_C, alpha), run_time=0.35)

        note = mono("acceleration comes from: tiled data reuse + mixed precision + dedicated matrix units",
                    0.31, YELLOW_E).next_to(title, DOWN, buff=0.25)
        self.play(FadeIn(note))
        self.wait(1.2)


# ===========================================================================
class Roofline(Scene):
    """Roofline: tensor cores add a much higher compute roof for GEMM."""

    def construct(self):
        title = mono("roofline: data reuse gets you to the hardware roof", 0.46, WHITE, BOLD)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title))

        ax = Axes(x_range=[-1, 3.6, 1], y_range=[0, 4.4, 1], x_length=9, y_length=5.2,
                  axis_config={"include_numbers": False, "stroke_width": 2}).shift(DOWN * 0.4)
        xl = mono("arithmetic intensity  (FLOP/byte, log)", 0.32).next_to(ax, DOWN, buff=0.3)
        yl = mono("throughput (log)", 0.32).rotate(PI / 2).next_to(ax, LEFT, buff=0.2)
        self.play(Create(ax), Write(xl), Write(yl))

        mem = ax.plot(lambda x: min(x + 2.3, 2.0), x_range=[-1, 3.6], color=BLUE_C,
                      stroke_width=3)
        simd = DashedLine(ax.c2p(-1, 2.0), ax.c2p(3.6, 2.0), color=GREEN_C, stroke_width=3)
        tensor_roof = DashedLine(ax.c2p(-1, 3.65), ax.c2p(3.6, 3.65), color=TC_C, stroke_width=4)
        mem_l = mono("memory bandwidth roof", 0.28, BLUE_C).next_to(ax.c2p(-0.35, 1.9), UL, buff=0.05)
        simd_l = mono("SIMD/FMA roof", 0.28, GREEN_C).next_to(ax.c2p(2.2, 2.0), UP, buff=0.1)
        tc_l = mono("tensor core roof", 0.28, TC_C, BOLD).next_to(ax.c2p(2.2, 3.65), UP, buff=0.1)
        self.play(Create(mem), Create(simd), Create(tensor_roof), Write(mem_l), Write(simd_l), Write(tc_l))

        pts = [(-0.7, 0.2, "naive", RED),
               (0.2, 1.3, "blocked", YELLOW_E),
               (0.9, 1.8, "NEON", ORANGE),
               (2.3, 3.1, "CUDA tiled", TEAL_B),
               (3.05, 3.55, "tensor cores", TC_C)]
        dots, labels = VGroup(), VGroup()
        for x, y, t, c in pts:
            d = Dot(ax.c2p(x, y), color=c, radius=0.09)
            lab = mono(t, 0.27, c).next_to(d, RIGHT, buff=0.12)
            dots.add(d); labels.add(lab)
        for i in range(len(pts)):
            self.play(GrowFromCenter(dots[i]), FadeIn(labels[i]), run_time=0.45)

        arrow = Arrow(ax.c2p(-0.7, 0.2), ax.c2p(3.05, 3.55), buff=0.15, color=WHITE,
                      stroke_width=3)
        note = mono("same math: C=A*B+C\nbetter schedule + special matrix units", 0.31, GREY_A) \
            .move_to(ax.c2p(0.7, 0.75))
        self.play(GrowArrow(arrow), FadeIn(note))
        self.wait(1.2)
