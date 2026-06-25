"""The Convolution Zoo — Manim scenes (Manim Community).

Every convolution variant, how they all collapse to GEMM, how GEMM runs on
NEON / Arm SME / NVIDIA Tensor Cores, and why that powers GenAI.

Render all:
  manim -qh --format mp4 convzoo_anim.py ColdOpen SpatialFamily GemmUnify \
        ChannelZoo ShapeTricks NEON SME TensorCore WhyMatrixWins GenAI
"""
from manim import *

# Color = role (see director.md)
IN_C, W_C, OUT_C, HL = BLUE_B, YELLOW_E, GREEN_B, ORANGE
TC_C, MEM_C, ACC_C = "#B67CFF", "#4D8DFF", "#42E8D8"


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


def title(s, scale=0.5):
    return mono(s, scale, WHITE, BOLD).to_edge(UP, buff=0.4)


def cube(rows, cols, depth=3, cell=0.32, off=0.16, color=IN_C):
    """Fake 3D: depth stacked grids, drawn back-to-front."""
    planes = []
    for d in range(depth):
        g = grid(rows, cols, cell, color=color)
        for sq in g:
            sq.set_fill(color, 0.18)
        g.shift(RIGHT * off * d + UP * off * d)
        planes.append(g)
    return VGroup(*reversed(planes)), planes


# ===========================================================================
class ColdOpen(Scene):
    """Hook: every AI creation is a convolution; one math, a dozen disguises."""

    def construct(self):
        t = mono("THE CONVOLUTION ZOO", 0.95, WHITE, BOLD).shift(UP * 2.4)
        sub = mono("the one operation behind every image, video & voice AI makes",
                   0.4, GREY_B).next_to(t, DOWN, buff=0.35)
        self.play(Write(t), run_time=1.0)
        self.play(FadeIn(sub, shift=UP * 0.2))

        # a row of variant "chips" — the zoo
        names = ["1D", "2D", "3D", "1x1", "grouped", "depthwise", "dilated", "transposed"]
        cols = [IN_C, IN_C, IN_C, W_C, OUT_C, OUT_C, HL, TC_C]
        chips = VGroup()
        for nm, c in zip(names, cols):
            box = VGroup(RoundedRectangle(width=1.5, height=0.7, corner_radius=0.12,
                                          stroke_color=c, fill_color=c, fill_opacity=0.16),
                         mono(nm, 0.34, c))
            chips.add(box)
        chips.arrange_in_grid(rows=2, cols=4, buff=0.3).shift(DOWN * 0.7)
        self.play(LaggedStart(*[FadeIn(c, scale=0.7) for c in chips],
                              lag_ratio=0.12, run_time=2.0))
        self.wait(0.4)

        punch = mono("same math.  a dozen disguises.", 0.5, HL, BOLD).to_edge(DOWN, buff=0.7)
        self.play(Write(punch))
        self.play(Indicate(punch, color=HL, scale_factor=1.05))
        self.wait(0.8)


# ===========================================================================
class SpatialFamily(Scene):
    """1D / 2D / 3D: the same slide-and-sum, one more axis each time."""

    def construct(self):
        self.play(Write(title("the spatial family: slide a kernel, take a weighted sum")))

        # 1D
        g1 = grid(1, 9, 0.34, IN_C).shift(LEFT * 4.3 + UP * 1.2)
        l1 = mono("1D  ·  audio", 0.34, IN_C).next_to(g1, UP, buff=0.25)
        # 2D
        g2 = grid(6, 6, 0.34, IN_C).shift(UP * 1.0)
        l2 = mono("2D  ·  images", 0.34, IN_C).next_to(g2, UP, buff=0.2)
        # 3D
        c3, planes3 = cube(5, 5, 3, 0.3, 0.16, IN_C)
        c3.shift(RIGHT * 4.2 + UP * 1.0)
        l3 = mono("3D  ·  video", 0.34, IN_C).next_to(c3, UP, buff=0.2)
        self.play(FadeIn(g1), FadeIn(g2), FadeIn(c3), Write(l1), Write(l2), Write(l3))

        # sliding kernels
        k1 = SurroundingRectangle(VGroup(g1.cellat(0, 0), g1.cellat(0, 2)),
                                  color=W_C, stroke_width=4, buff=0.0)
        k2 = SurroundingRectangle(VGroup(g2.cellat(0, 0), g2.cellat(2, 2)),
                                  color=W_C, stroke_width=4, buff=0.0)
        k3 = SurroundingRectangle(VGroup(planes3[0].cellat(0, 0), planes3[0].cellat(2, 2)),
                                  color=W_C, stroke_width=4, buff=0.0)
        self.add(k1, k2, k3)
        for step in range(4):
            self.play(
                k1.animate.move_to(VGroup(g1.cellat(0, step + 1), g1.cellat(0, step + 3)).get_center()),
                k2.animate.move_to(VGroup(g2.cellat(min(step, 3), min(step, 3)),
                                          g2.cellat(min(step, 3) + 2, min(step, 3) + 2)).get_center()),
                run_time=0.45)

        note = mono("one more axis each time — and the multiplies explode (3x3x3 over video!)",
                    0.36, GREY_A).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(note))
        self.wait(0.8)


# ===========================================================================
class GemmUnify(Scene):
    """im2col: every sliding window becomes one matrix multiply."""

    def construct(self):
        self.play(Write(title("the unifying trick: every convolution IS a matrix multiply")))

        inp = grid(5, 5, 0.42, IN_C).to_edge(LEFT, buff=0.9).shift(DOWN * 0.2)
        il = mono("input", 0.34, IN_C).next_to(inp, UP, buff=0.2)
        col = grid(9, 9, 0.3, GREY_B).to_edge(RIGHT, buff=0.9).shift(DOWN * 0.2)
        cl = mono("im2col matrix", 0.34).next_to(col, UP, buff=0.2)
        self.play(FadeIn(inp), Write(il), FadeIn(col), Write(cl))

        pal = color_gradient([BLUE_D, TEAL, GREEN, YELLOW, RED], 9)
        K = 3
        for p in (0, 1):
            pr, pc = p // 3, p % 3
            win = SurroundingRectangle(VGroup(inp.cellat(pr, pc), inp.cellat(pr + K - 1, pc + K - 1)),
                                       color=HL, stroke_width=4, buff=0.0)
            self.play(Create(win), run_time=0.4)
            cps = VGroup()
            for ky in range(3):
                for kx in range(3):
                    idx = ky * 3 + kx
                    cp = inp.cellat(pr + ky, pc + kx).copy().set_fill(pal[idx], 0.95)
                    cps.add(cp)
            self.add(cps)
            self.play(*[cps[i].animate.move_to(col.cellat(i, p).get_center()).scale(0.3 / 0.42)
                        for i in range(9)], run_time=0.7)
            for i in range(9):
                col.cellat(i, p).set_fill(pal[i], 0.95)
            self.remove(cps); self.play(FadeOut(win), run_time=0.2)
        # fill rest
        self.play(*[col.cellat(i, p).animate.set_fill(pal[i], 0.8)
                    for p in range(2, 9) for i in range(9)], run_time=0.5)

        eq = mono("conv  =  W · X   (one GEMM)", 0.5, HL, BOLD).to_edge(DOWN, buff=1.0)
        q = mono("so: don't make conv fast — make MATRIX MULTIPLY fast", 0.38, GREEN_B) \
            .next_to(eq, DOWN, buff=0.2)
        self.play(Write(eq)); self.play(FadeIn(q))
        self.wait(0.9)


# ===========================================================================
class ChannelZoo(Scene):
    """1x1, grouped, depthwise, separable — playing with channels."""

    def construct(self):
        self.play(Write(title("the channel zoo: play with channels, change everything")))

        def stack(n, x, y, col=IN_C, cell=0.26):
            g = VGroup()
            for i in range(n):
                s = Square(cell, stroke_width=1.2, stroke_color=col,
                           fill_color=col, fill_opacity=0.3).shift(RIGHT * 0.12 * i + UP * 0.12 * i)
                g.add(s)
            return g.move_to([x, y, 0])

        # 1x1
        a = stack(5, -5, 1.4, W_C)
        al = mono("1x1\npure matmul", 0.3, W_C).next_to(a, DOWN, buff=0.35)
        # grouped
        b = VGroup(stack(2, -1.9, 1.55, OUT_C), stack(2, -1.5, 1.2, GREEN_C)).move_to([-1.7, 1.4, 0])
        bl = mono("grouped\nsplit channels", 0.3, OUT_C).next_to(b, DOWN, buff=0.35)
        # depthwise
        c = VGroup(*[Square(0.3, stroke_color=HL, fill_color=HL, fill_opacity=0.3).shift(RIGHT * 0.42 * i)
                     for i in range(4)]).move_to([1.6, 1.4, 0])
        cl = mono("depthwise\n1 filter / channel", 0.3, HL).next_to(c, DOWN, buff=0.35)
        # separable
        d = VGroup(mono("DW", 0.3, HL), mono("+", 0.3, WHITE), mono("1x1", 0.3, W_C)).arrange(RIGHT, buff=0.2).move_to([4.6, 1.4, 0])
        dl = mono("separable\n(MobileNet)", 0.3, GREEN_B).next_to(d, DOWN, buff=0.35)

        for obj, lab in [(a, al), (b, bl), (c, cl), (d, dl)]:
            self.play(FadeIn(obj, scale=0.8), Write(lab), run_time=0.6)

        # depthwise is the rebel
        reb = mono("depthwise can't become one GEMM — no channels to contract",
                   0.34, HL).shift(DOWN * 1.4)
        self.play(FadeIn(reb))
        win = mono("separable: ~9x fewer multiplies, same job", 0.42, GREEN_B, BOLD).to_edge(DOWN, buff=0.7)
        self.play(Write(win), Indicate(win, color=GREEN_B, scale_factor=1.05))
        self.wait(0.8)


# ===========================================================================
class ShapeTricks(Scene):
    """Dilated (reach far) and transposed (grow the image)."""

    def construct(self):
        self.play(Write(title("two more shapes, two more powers")))

        # dilated — taps with gaps
        row = grid(1, 11, 0.42, IN_C).shift(LEFT * 3.0 + UP * 1.1)
        dl = mono("dilated: gaps let the kernel see far  (WaveNet)", 0.34, HL) \
            .next_to(row, UP, buff=0.3)
        self.play(FadeIn(row), Write(dl))
        taps = [0, 2, 4]  # dilation 2
        self.play(*[row.cellat(0, t).animate.set_fill(W_C, 0.9) for t in taps])
        for shift in range(3):
            new = [t + shift + 1 for t in taps]
            if new[-1] >= 11:
                break
            self.play(*[row.cellat(0, t).animate.set_fill(IN_C, 0.0) for t in taps],
                      *[row.cellat(0, t).animate.set_fill(W_C, 0.9) for t in new], run_time=0.4)
            taps = new

        # transposed — small -> large
        small = grid(2, 2, 0.5, IN_C).shift(LEFT * 2.0 + DOWN * 1.7)
        for s in small:
            s.set_fill(IN_C, 0.4)
        big = grid(4, 4, 0.5, OUT_C).shift(RIGHT * 2.2 + DOWN * 1.5)
        tl = mono("transposed: grow the image — learnable upsampling (generators)",
                  0.34, TC_C).to_edge(DOWN, buff=0.7)
        arr = Arrow(small.get_right(), big.get_left(), buff=0.3, color=TC_C, stroke_width=4)
        self.play(FadeIn(small))
        self.play(GrowArrow(arr), LaggedStart(*[c.animate.set_fill(OUT_C, 0.5) for c in big],
                                              lag_ratio=0.05, run_time=1.0))
        self.play(Write(tl))
        self.wait(0.8)


# ===========================================================================
class NEON(Scene):
    """SIMD: broadcast one weight, FMA across 4 lanes. Reuse along 1 axis."""

    def construct(self):
        self.play(Write(title("NEON: a CPU vector unit — 4 numbers at once")))

        w = VGroup(Square(0.7, stroke_color=W_C, fill_color=W_C, fill_opacity=0.6),
                   mono("w", 0.5, BLACK, BOLD)).shift(LEFT * 4.5 + UP * 1.4)
        lanes = VGroup(*[Square(0.7, stroke_color=W_C, fill_color=W_C, fill_opacity=0.5)
                         for _ in range(4)]).arrange(RIGHT, buff=0.1).shift(UP * 1.4 + RIGHT * 0.5)
        self.play(FadeIn(w))
        self.play(LaggedStart(*[TransformFromCopy(w[0], l) for l in lanes], lag_ratio=0.1),
                  Write(mono("broadcast", 0.34, W_C).next_to(lanes, UP, buff=0.2)))

        inp = grid(1, 4, 0.7, IN_C).shift(DOWN * 0.2 + RIGHT * 0.5)
        for c in range(4):
            inp.cellat(0, c).set_fill(IN_C, 0.4)
        acc = grid(1, 4, 0.7, OUT_C).shift(DOWN * 2.2 + RIGHT * 0.5)
        self.play(FadeIn(inp), Write(mono("4 inputs (vld1q)", 0.3, IN_C).next_to(inp, LEFT, buff=0.3)),
                  FadeIn(acc), Write(mono("4 outputs", 0.3, OUT_C).next_to(acc, LEFT, buff=0.3)))
        arrows = VGroup(*[Arrow(inp.cellat(0, i).get_bottom(), acc.cellat(0, i).get_top(),
                                buff=0.1, color=HL, stroke_width=4) for i in range(4)])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.0, run_time=0.6),
                  *[acc.cellat(0, i).animate.set_fill(OUT_C, 0.85) for i in range(4)])
        self.play(Write(mono("vfmaq: acc += in * w", 0.4, HL).shift(DOWN * 3.4 + RIGHT * 0.5)))
        lim = mono("reuse along ONE axis: load 4, do 4 multiplies", 0.38, RED).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(lim))
        self.wait(0.8)


# ===========================================================================
class SME(Scene):
    """SME: outer product of a column and row fills a 2D ZA tile in one op."""

    def construct(self):
        self.play(Write(title("Arm SME: a 2D matrix accumulator on chip (Apple M4)")))

        n = 6
        za = grid(n, n, 0.46, ACC_C).shift(DOWN * 0.3 + RIGHT * 0.8)
        zal = mono("ZA tile", 0.34, ACC_C).next_to(za, RIGHT, buff=0.3)
        colv = grid(n, 1, 0.46, IN_C).next_to(za, LEFT, buff=0.9)
        for s in colv:
            s.set_fill(IN_C, 0.5)
        rowv = grid(1, n, 0.46, W_C).next_to(za, UP, buff=0.5).align_to(za, LEFT)
        for s in rowv:
            s.set_fill(W_C, 0.5)
        self.play(FadeIn(za), Write(zal), FadeIn(colv), FadeIn(rowv),
                  Write(mono("column", 0.3, IN_C).next_to(colv, LEFT, buff=0.2)),
                  Write(mono("row", 0.3, W_C).next_to(rowv, UP, buff=0.15)))

        eq = mono("FMOPA:  ZA += column ⊗ row   (outer product)", 0.42, HL, BOLD).to_edge(DOWN, buff=0.9)
        self.play(Write(eq))
        # fill the whole tile = every col[i] * row[j]
        anims = []
        for i in range(n):
            for j in range(n):
                anims.append(za.cellat(i, j).animate.set_fill(ACC_C, 0.85))
        self.play(LaggedStart(*anims, lag_ratio=0.012, run_time=1.6))
        self.play(Indicate(za, color=ACC_C, scale_factor=1.03))
        note = mono("ONE instruction → the whole tile.  a vector reused across a full dimension.",
                    0.36, GREEN_B).next_to(eq, UP, buff=0.25)
        self.play(FadeIn(note))
        self.wait(0.8)


# ===========================================================================
class TensorCore(Scene):
    """NVIDIA Tensor Core: warp-level MMA, fragments, mixed precision."""

    def construct(self):
        self.play(Write(title("NVIDIA Tensor Cores: a whole warp computes one tile")))

        mem = VGroup(
            RoundedRectangle(width=2.3, height=0.7, corner_radius=0.1, color=MEM_C, stroke_width=3),
            mono("HBM → shared → registers", 0.26, MEM_C))
        mem.shift(LEFT * 4.4 + UP * 1.9)
        self.play(Create(mem[0]), FadeIn(mem[1]))

        Afrag = grid(4, 4, 0.26, IN_C, IN_C, 0.18).shift(LEFT * 1.3 + UP * 1.4)
        Bfrag = grid(4, 4, 0.26, W_C, W_C, 0.18).shift(RIGHT * 1.0 + UP * 1.4)
        tc = RoundedRectangle(width=2.6, height=1.2, corner_radius=0.16, color=TC_C,
                              fill_color=TC_C, fill_opacity=0.15, stroke_width=4).shift(DOWN * 0.4)
        tcl = mono("Tensor Core\nMMA", 0.32, TC_C, BOLD).move_to(tc)
        Dfrag = grid(4, 4, 0.26, OUT_C, OUT_C, 0.1).shift(RIGHT * 3.1 + DOWN * 1.7)
        self.play(FadeIn(Afrag), FadeIn(Bfrag), Create(tc), FadeIn(tcl), FadeIn(Dfrag),
                  Write(mono("A frag", 0.26, IN_C).next_to(Afrag, UP, buff=0.1)),
                  Write(mono("B frag", 0.26, W_C).next_to(Bfrag, UP, buff=0.1)))
        a1 = Arrow(Afrag.get_bottom(), tc.get_left(), buff=0.1, color=IN_C)
        a2 = Arrow(Bfrag.get_bottom(), tc.get_top(), buff=0.1, color=W_C)
        a3 = Arrow(tc.get_right(), Dfrag.get_left(), buff=0.1, color=OUT_C)
        self.play(GrowArrow(a1), GrowArrow(a2), GrowArrow(a3))

        eq = mono("D = A · B + C   —  thousands of MACs / instruction", 0.42, WHITE, BOLD).to_edge(DOWN, buff=0.85)
        mp = mono("mixed precision: fp16 multiply, fp32 accumulate", 0.34, ACC_C).next_to(eq, UP, buff=0.2)
        self.play(Write(eq), FadeIn(mp))
        for a in (0.4, 0.7, 0.95):
            self.play(Afrag.animate.set_fill(IN_C, a), Bfrag.animate.set_fill(W_C, a),
                      Indicate(tc, color=TC_C, scale_factor=1.05),
                      Dfrag.animate.set_fill(OUT_C, a), run_time=0.5)
        self.wait(0.7)


# ===========================================================================
class WhyMatrixWins(Scene):
    """O(L) vs O(L^2) reuse, then the 3-roof roofline."""

    def construct(self):
        self.play(Write(title("why matrix hardware wins: L  vs  L²")))

        # left: SIMD vector -> L MACs
        vec = grid(1, 5, 0.4, IN_C).shift(LEFT * 4.0 + UP * 1.7)
        for s in vec:
            s.set_fill(IN_C, 0.5)
        sl = mono("SIMD: vector → L MACs", 0.32, IN_C).next_to(vec, UP, buff=0.2)
        outg = grid(5, 5, 0.34, ACC_C).shift(LEFT * 4.0 + DOWN * 1.4)
        ol = mono("matrix: outer product → L² MACs", 0.32, ACC_C).next_to(outg, DOWN, buff=0.2)
        self.play(FadeIn(vec), Write(sl))
        self.play(*[s.animate.set_fill(IN_C, 0.85) for s in vec], run_time=0.5)
        self.play(FadeIn(outg), Write(ol))
        self.play(LaggedStart(*[c.animate.set_fill(ACC_C, 0.85) for c in outg],
                              lag_ratio=0.01, run_time=1.2))
        big = mono("same memory traffic.\nquadratically more math.", 0.4, HL, BOLD).shift(LEFT * 4.0 + DOWN * 3.2)
        self.play(FadeIn(big))

        # right: mini roofline with 3 roofs
        ax = Axes(x_range=[-1, 3, 1], y_range=[0, 4, 1], x_length=5.2, y_length=4.4,
                  axis_config={"include_numbers": False, "stroke_width": 2}).shift(RIGHT * 3.0 + DOWN * 0.3)
        mem = ax.plot(lambda x: min(x + 2.4, 2.0), x_range=[-1, 3], color=MEM_C, stroke_width=3)
        simd = DashedLine(ax.c2p(-1, 2.0), ax.c2p(3, 2.0), color=GREEN_C, stroke_width=3)
        mat = DashedLine(ax.c2p(-1, 3.4), ax.c2p(3, 3.4), color=TC_C, stroke_width=4)
        self.play(Create(ax), Create(mem), Create(simd), Create(mat),
                  Write(mono("matrix roof", 0.26, TC_C).next_to(ax.c2p(1.2, 3.4), UP, buff=0.05)),
                  Write(mono("SIMD roof", 0.26, GREEN_C).next_to(ax.c2p(1.4, 2.0), UP, buff=0.05)))
        arrow = Arrow(ax.c2p(-0.7, 0.3), ax.c2p(1.6, 3.3), buff=0.1, color=WHITE, stroke_width=3)
        self.play(GrowArrow(arrow))
        self.wait(0.8)


# ===========================================================================
class GenAI(Scene):
    """Where the zoo lives today -> all lowered to GEMM on matrix engines."""

    def construct(self):
        self.play(Write(title("where the zoo lives: this is what powers GenAI")))

        rows = [
            ("Diffusion U-Net / VAE", "3x3 + 1x1 convs", IN_C),
            ("DiT / SD3 / FLUX", "patchify = strided conv", W_C),
            ("Video (Sora-like)", "causal 3D convs", OUT_C),
            ("Audio (EnCodec, Whisper)", "1D convs", HL),
            ("LLMs (Mamba)", "depthwise conv1d, k=4", TC_C),
        ]
        items = VGroup()
        for name, kind, c in rows:
            box = VGroup(
                RoundedRectangle(width=4.6, height=0.62, corner_radius=0.1,
                                 stroke_color=c, fill_color=c, fill_opacity=0.12),
                VGroup(mono(name, 0.3, WHITE), mono(kind, 0.26, c)).arrange(RIGHT, buff=0.35))
            box[1].move_to(box[0])
            items.add(box)
        items.arrange(DOWN, buff=0.22).to_edge(LEFT, buff=0.8).shift(DOWN * 0.2)
        self.play(LaggedStart(*[FadeIn(it, shift=RIGHT * 0.2) for it in items],
                              lag_ratio=0.18, run_time=2.2))

        gemm = VGroup(RoundedRectangle(width=2.5, height=1.0, corner_radius=0.14,
                                       color=GREEN_B, fill_color=GREEN_B, fill_opacity=0.16),
                      mono("GEMM", 0.42, GREEN_B, BOLD)).to_edge(RIGHT, buff=2.6).shift(UP * 0.7)
        tcore = VGroup(RoundedRectangle(width=2.5, height=1.0, corner_radius=0.14,
                                        color=TC_C, fill_color=TC_C, fill_opacity=0.16),
                       mono("Tensor\nCores", 0.36, TC_C, BOLD)).next_to(gemm, DOWN, buff=0.9)
        self.play(FadeIn(gemm), FadeIn(tcore))
        arrs = VGroup(*[Arrow(it.get_right(), gemm.get_left(), buff=0.15,
                              color=GREY_B, stroke_width=2.5,
                              max_tip_length_to_length_ratio=0.05) for it in items])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrs], lag_ratio=0.08, run_time=1.0))
        self.play(GrowArrow(Arrow(gemm.get_bottom(), tcore.get_top(), buff=0.1,
                                  color=GREEN_B, stroke_width=4)))

        punch = mono("same math, a dozen disguises — feeding the engines of generative AI",
                     0.4, HL, BOLD).to_edge(DOWN, buff=0.6)
        self.play(Write(punch), Indicate(punch, color=HL, scale_factor=1.03))
        self.wait(1.2)
