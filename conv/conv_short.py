"""Vertical (9:16) YouTube Short: 'conv2d, 14x faster with NEON' in ~45s.

Portrait canvas: 1080x1920, with an 8 wide x 14.22 tall coordinate frame.

Render:
  .../manim-ce/bin/manim -qh --format mp4 conv_short.py Short
"""
from manim import *

# --- portrait configuration (must match 1080:1920 = 0.5625) ---
config.pixel_width = 1080
config.pixel_height = 1920
config.frame_width = 8.0
config.frame_height = 14.222

IN_C, OUT_C, K_C, HL = BLUE_B, GREEN_B, YELLOW_E, ORANGE


def grid(rows, cols, cell=0.5, color=GREY_B, fill=BLACK, fop=0.0):
    g = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(cell, stroke_width=1.4, stroke_color=color,
                        fill_color=fill, fill_opacity=fop)
            sq.move_to([c * cell, -r * cell, 0])
            g.add(sq)
    g.cellat = lambda r, c: g[r * cols + c]
    g.cols = cols
    g.move_to(ORIGIN)
    return g


def mono(s, scale=0.5, color=WHITE, weight=NORMAL):
    return Text(s, font="Menlo", color=color, weight=weight).scale(scale)


class Short(Scene):
    def construct(self):
        # ---- beat 1: title (~0-6s) ----
        t1 = mono("conv2d", 1.3, WHITE, BOLD).shift(UP * 1.2)
        t2 = mono("14x faster", 1.0, HL, BOLD).next_to(t1, DOWN, buff=0.5)
        t3 = mono("with NEON SIMD", 0.55, GREY_B).next_to(t2, DOWN, buff=0.4)
        self.play(Write(t1), run_time=0.8)
        self.play(FadeIn(t2, shift=UP * 0.3), FadeIn(t3), run_time=0.8)
        self.wait(0.8)
        self.play(FadeOut(VGroup(t1, t2, t3)), run_time=0.5)

        # ---- beat 2: the naive 6 loops are slow (~6-15s) ----
        head = mono("the naive way", 0.62, WHITE, BOLD).to_edge(UP, buff=1.0)
        loops = VGroup(
            mono("for oc:", 0.5, GREY_B),
            mono("  for oy, ox:", 0.5, GREY_B),
            mono("    for ic, ky, kx:", 0.5, GREY_B),
            mono("      acc += in * w", 0.5, HL),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).shift(UP * 3.2)
        slow = mono("1 pixel at a time = slow", 0.5, RED).next_to(loops, DOWN, buff=0.6)
        self.play(Write(head))
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in loops],
                              lag_ratio=0.25, run_time=1.6))
        self.play(FadeIn(slow))
        self.wait(1.0)
        self.play(FadeOut(VGroup(head, loops, slow)), run_time=0.5)

        # ---- beat 3: the trick - broadcast weight, FMA across 4 lanes (~15-32s) ----
        head2 = mono("vectorize across width", 0.6, WHITE, BOLD).to_edge(UP, buff=0.9)
        self.play(Write(head2))

        # one weight, broadcast to 4 lanes
        wcell = VGroup(Square(0.7, stroke_color=K_C, fill_color=K_C, fill_opacity=0.6),
                       mono("w", 0.5, BLACK, BOLD)).shift(UP * 4.6)
        self.play(FadeIn(wcell, scale=0.6))

        lanes = VGroup(*[Square(0.7, stroke_color=K_C, fill_color=K_C, fill_opacity=0.5)
                         for _ in range(4)]).arrange(RIGHT, buff=0.12).shift(UP * 2.7)
        splat = mono("broadcast -> 4 lanes", 0.42, K_C).next_to(lanes, DOWN, buff=0.25)
        self.play(LaggedStart(*[TransformFromCopy(wcell[0], l) for l in lanes],
                              lag_ratio=0.1, run_time=1.0), Write(splat))

        # input row (contiguous) and accumulators
        inrow = grid(1, 4, 0.7, color=IN_C).shift(UP * 0.3)
        for c in range(4):
            inrow.cellat(0, c).set_fill(IN_C, 0.35)
        inlab = mono("4 input pixels (vld1q)", 0.4, IN_C).next_to(inrow, UP, buff=0.2)
        acc = grid(1, 4, 0.7, color=OUT_C).shift(DOWN * 2.2)
        acclab = mono("4 outputs (acc)", 0.4, OUT_C).next_to(acc, DOWN, buff=0.2)
        self.play(FadeIn(inrow), FadeIn(inlab), FadeIn(acc), FadeIn(acclab))

        # FMA: lanes * input -> acc
        arrows = VGroup(*[Arrow(inrow.cellat(0, i).get_bottom(),
                                acc.cellat(0, i).get_top(), buff=0.1,
                                stroke_width=4, color=HL,
                                max_tip_length_to_length_ratio=0.2) for i in range(4)])
        fma = mono("vfmaq_f32: acc += in * w", 0.44, HL).shift(DOWN * 4.0)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.0, run_time=0.6),
                  *[acc.cellat(0, i).animate.set_fill(OUT_C, 0.85) for i in range(4)],
                  Indicate(lanes, color=YELLOW, scale_factor=1.1))
        self.play(Write(fma))
        self.wait(0.8)

        # 16 in flight
        flight = mono("x16 in flight -> no stalls", 0.5, WHITE, BOLD).shift(DOWN * 5.2)
        self.play(FadeIn(flight, scale=1.1))
        self.wait(1.0)
        self.play(FadeOut(VGroup(head2, wcell, lanes, splat, inrow, inlab,
                                 acc, acclab, arrows, fma, flight)), run_time=0.5)

        # ---- beat 4: payoff (~32-45s) ----
        p1 = mono("same math", 0.6, GREY_B).shift(UP * 2.0)
        p2 = mono("bit-for-bit identical", 0.55, GREEN_B).next_to(p1, DOWN, buff=0.4)
        big = mono("14x", 2.4, HL, BOLD).shift(DOWN * 0.6)
        p3 = mono("on Apple M1 Pro", 0.5, GREY_B).next_to(big, DOWN, buff=0.5)
        self.play(FadeIn(p1), FadeIn(p2))
        self.play(Write(big), run_time=0.8)
        self.play(FadeIn(p3))
        # speed bars
        bar_n = Rectangle(width=1.2, height=0.5, fill_color=GREY, fill_opacity=0.8,
                          stroke_width=0).shift(DOWN * 3.6 + LEFT * 1.5)
        bar_v = Rectangle(width=1.2, height=0.5, fill_color=HL, fill_opacity=0.9,
                          stroke_width=0).shift(DOWN * 4.4 + LEFT * 1.5)
        self.play(FadeIn(bar_n))
        self.play(bar_v.animate.stretch_to_fit_width(5.6, about_edge=LEFT), run_time=0.9)
        self.wait(1.2)
