"""Manim visualization of the conv2d test bed.

Renders two scenes that mirror the C loops in conv2d_naive.c / conv2d_neon.c:

  NaiveConv  -> one output pixel at a time: the 6-deep loop nest.
  NeonConv   -> the optimized idea: broadcast a weight, FMA across the
                output-WIDTH dimension (4 lanes / 16 pixels) at once.

Render:
  .venv/bin/manim -pql conv_anim.py NaiveConv
  .venv/bin/manim -pqh conv_anim.py NeonConv      # high quality
"""
from manim import *

# ------- small geometry helpers -------------------------------------------
IN_C, K_C, OUT_C, HL = BLUE_E, YELLOW_E, GREEN_E, ORANGE


def grid(rows, cols, cell=0.52, color=GREY_B, fill=BLACK):
    g = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(cell, stroke_width=1.4, stroke_color=color,
                        fill_color=fill, fill_opacity=0.0)
            sq.move_to([c * cell, -r * cell, 0])
            g.add(sq)
    g.rows, g.cols, g.cell = rows, cols, cell
    g.cellat = lambda r, c: g[r * cols + c]
    g.move_to(ORIGIN)
    return g


def label(txt, obj, scale=0.5, color=WHITE):
    return Text(txt, font="Menlo", color=color).scale(scale).next_to(obj, UP, buff=0.2)


# ===========================================================================
class NaiveConv(Scene):
    """Visualize conv2d_naive: the textbook 6-deep loop nest, one pixel at a time."""

    def construct(self):
        H = W = 7
        K = 3
        OH = OW = H - K + 1

        inp = grid(H, W, color=BLUE_B).to_edge(LEFT, buff=0.8).shift(UP * 0.3)
        out = grid(OH, OW, color=GREEN_B).to_edge(RIGHT, buff=1.2).shift(UP * 0.3)
        self.play(FadeIn(inp), FadeIn(out))
        self.play(Write(label("input  [H][W]", inp, color=IN_C)),
                  Write(label("output [OH][OW]", out, color=OUT_C)))

        # the 3x3 kernel window that slides over the input
        win = SurroundingRectangle(
            VGroup(inp.cellat(0, 0), inp.cellat(K - 1, K - 1)),
            color=K_C, stroke_width=5, buff=0.0)

        # the live "acc" register being summed for one output pixel
        code = Text("for oc:  for oy:  for ox:        # one output pixel",
                    font="Menlo").scale(0.42).to_edge(DOWN, buff=1.1)
        inner = Text("  acc = bias;  for ic,ky,kx:  acc += in*w",
                     font="Menlo", color=HL).scale(0.42).next_to(code, DOWN, buff=0.15)
        self.play(Write(code), Write(inner))

        self.add(win)
        # LOOP over output pixels (oy, ox): slide the window, light the target.
        order = [(0, 0), (0, 1), (0, 2), (1, 0)]  # a few representative pixels
        for (oy, ox) in order:
            tgt = out.cellat(oy, ox)
            self.play(
                win.animate.move_to(
                    VGroup(inp.cellat(oy, ox), inp.cellat(oy + K - 1, ox + K - 1))
                    .get_center()),
                tgt.animate.set_fill(OUT_C, 0.25),
                run_time=0.55)

            # inner reduction: flash each of the 9 (ky,kx) MACs into this pixel
            macs = VGroup()
            for ky in range(K):
                for kx in range(K):
                    macs.add(inp.cellat(oy + ky, ox + kx))
            self.play(LaggedStart(
                *[Indicate(m, color=K_C, scale_factor=1.25) for m in macs],
                lag_ratio=0.18, run_time=1.1))
            self.play(tgt.animate.set_fill(OUT_C, 0.9), run_time=0.25)

        note = Text("every weight re-read for every pixel — no reuse, no SIMD",
                    color=GREY_A).scale(0.4).next_to(out, DOWN, buff=0.5)
        self.play(FadeIn(note))
        self.wait(1.2)


# ===========================================================================
class NeonConv(Scene):
    """Visualize conv2d_neon: broadcast one weight, FMA across output WIDTH."""

    def construct(self):
        H, W, K = 6, 12, 3
        OW = W - K + 1

        title = Text("conv2d_neon : vectorize across output WIDTH",
                     font="Menlo").scale(0.5).to_edge(UP, buff=0.4)
        self.play(Write(title))

        inp = grid(H, W, cell=0.5, color=BLUE_B).shift(UP * 1.1)
        out = grid(H - K + 1, OW, cell=0.5, color=GREEN_B).shift(DOWN * 1.7)
        self.play(FadeIn(inp), FadeIn(out))

        # one broadcast weight register (all 4 lanes identical)
        wbox = VGroup(*[Square(0.45, stroke_color=YELLOW, fill_color=YELLOW_E,
                               fill_opacity=0.5) for _ in range(4)]).arrange(RIGHT, buff=0.05)
        wbox.to_edge(LEFT, buff=0.6).shift(UP * 1.1)
        wlab = Text("w (splat)", font="Menlo", color=K_C).scale(0.4).next_to(wbox, UP, buff=0.15)
        self.play(FadeIn(wbox), Write(wlab))

        oy = 1  # the output row we're computing
        LANES = 4  # NEON float32x4_t

        fma = Text("a = vfmaq_f32(a, vld1q(row+kx), splat(w[kx]))",
                   font="Menlo", color=HL).scale(0.42).to_edge(DOWN, buff=0.35)
        self.play(Write(fma))

        # accumulator: 4 output lanes held live in registers
        acc_cells = VGroup(*[out.cellat(oy, c) for c in range(LANES)])
        self.play(acc_cells.animate.set_fill(OUT_C, 0.18),
                  Write(Text("acc lanes (ox..ox+3)", font="Menlo", color=OUT_C)
                        .scale(0.36).next_to(acc_cells, DOWN, buff=0.12)))

        # LOOP 6 (kx): slide the 4-wide vector load, FMA the broadcast weight.
        for kx in range(K):
            ky = 1  # show one kernel row for clarity
            load = VGroup(*[inp.cellat(oy + ky, c + kx) for c in range(LANES)])
            box = SurroundingRectangle(load, color=BLUE_B, stroke_width=4, buff=0.02)
            tag = Text(f"vld1q  kx={kx}", font="Menlo", color=BLUE_B).scale(0.36) \
                .next_to(box, UP, buff=0.1)
            self.play(Create(box), FadeIn(tag),
                      Indicate(wbox, color=YELLOW, scale_factor=1.1), run_time=0.6)

            # 4 parallel FMAs: load lane -> acc lane (same weight for all)
            arrows = VGroup(*[
                Arrow(load[i].get_bottom(), acc_cells[i].get_top(),
                      buff=0.05, stroke_width=3, color=HL, max_tip_length_to_length_ratio=0.15)
                for i in range(LANES)])
            self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.0,
                                  run_time=0.5),
                      acc_cells.animate.set_fill(OUT_C, 0.25 + 0.22 * (kx + 1)))
            self.play(FadeOut(box), FadeOut(tag), FadeOut(arrows), run_time=0.3)

        self.play(acc_cells.animate.set_fill(OUT_C, 0.95),
                  Indicate(acc_cells, color=GREEN, scale_factor=1.05))

        note = Text("4 lanes/instr  ·  x16 in the real loop hides FMA latency",
                    color=GREY_A).scale(0.4).next_to(out, UP, buff=0.25)
        self.play(FadeIn(note))
        self.wait(1.2)
