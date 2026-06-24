"""Manim visualization of multi-channel conv2d (e.g. a 3-channel RGB image).

Two axes of work that look the same in a loop nest but behave oppositely:

  ChannelReduce   -> Cin (input channels): every output pixel SUMS a partial
                     convolution from each input channel. This is a REDUCTION
                     (a dependency chain) -- it must accumulate.
  ChannelParallel -> Cout (output channels): each filter produces an entirely
                     independent output plane. This is PARALLEL -- map it across
                     threads / cores / SIMD lanes with no synchronization.

Render:
  .../manim-ce/bin/manim -ql --format mp4 conv_channels.py ChannelReduce
  .../manim-ce/bin/manim -ql --format mp4 conv_channels.py ChannelParallel
"""
from manim import *

# RGB-ish channel tints.
CH_COL = ["#e0666b", "#6aa84f", "#3d85c6"]   # R, G, B
CH_NAME = ["R", "G", "B"]
HL = ORANGE


def grid(rows, cols, cell=0.5, color=GREY_B, fill=BLACK, fop=0.0):
    g = VGroup()
    for r in range(rows):
        for c in range(cols):
            sq = Square(cell, stroke_width=1.2, stroke_color=color,
                        fill_color=fill, fill_opacity=fop)
            sq.move_to([c * cell, -r * cell, 0])
            g.add(sq)
    g.rows, g.cols, g.cell = rows, cols, cell
    g.cellat = lambda r, c: g[r * cols + c]
    g.move_to(ORIGIN)
    return g


def mtext(s, scale=0.42, color=WHITE):
    return Text(s, font="Menlo", color=color).scale(scale)


def channel_stack(n, rows, cols, cell=0.42, depth=0.22):
    """n grids stacked with a 2.5D depth offset; returns list front..back order."""
    planes = []
    for i in range(n):
        g = grid(rows, cols, cell, color=GREY_C)
        for sq in g:
            sq.set_fill(CH_COL[i], 0.30)
        g.shift(RIGHT * depth * i + UP * depth * i)   # depth illusion
        planes.append(g)
    # draw back-to-front so the front plane (i=0) sits on top
    group = VGroup(*reversed(planes))
    return planes, group


# ===========================================================================
class ChannelReduce(Scene):
    """One output pixel = sum of a partial conv from each of the 3 channels."""

    def construct(self):
        rows = cols = 5
        K = 3
        pr, pc = 1, 1     # the patch / output pixel we focus on

        title = mtext("multi-channel conv : SUM over input channels (Cin)", 0.48)
        title.to_edge(UP, buff=0.35)
        self.play(Write(title))

        planes, stack = channel_stack(3, rows, cols, 0.46, 0.26)
        stack.to_edge(LEFT, buff=1.0).shift(DOWN * 0.2)
        self.play(FadeIn(stack))
        # name each channel plane
        names = VGroup()
        for i, p in enumerate(planes):
            t = mtext(CH_NAME[i], 0.4, CH_COL[i]).next_to(p, UP, buff=0.05).shift(LEFT * 0.05)
            names.add(t)
        self.play(LaggedStart(*[Write(t) for t in names], lag_ratio=0.2))

        # a window over the SAME location in all three planes at once
        wins = VGroup()
        for p in planes:
            w = SurroundingRectangle(
                VGroup(p.cellat(pr, pc), p.cellat(pr + K - 1, pc + K - 1)),
                color=p[0].get_fill_color(), stroke_width=4, buff=0.0)
            wins.add(w)
        self.play(LaggedStart(*[Create(w) for w in wins], lag_ratio=0.2))

        # three partial products, one per channel, then summed into one pixel
        parts = VGroup()
        for i in range(3):
            b = VGroup(Square(0.62, fill_color=CH_COL[i], fill_opacity=0.55,
                              stroke_color=CH_COL[i]),
                       mtext(f"{CH_NAME[i]}*w", 0.32))
            parts.add(b)
        parts.arrange(DOWN, buff=0.45).move_to(RIGHT * 1.0 + DOWN * 0.2)
        plus = VGroup(mtext("+", 0.6), mtext("+", 0.6))
        plus[0].move_to((parts[0].get_center() + parts[1].get_center()) / 2)
        plus[1].move_to((parts[1].get_center() + parts[2].get_center()) / 2)

        for i in range(3):
            patch = VGroup(*[planes[i].cellat(pr + ky, pc + kx)
                             for ky in range(K) for kx in range(K)])
            self.play(Indicate(patch, color=WHITE, scale_factor=1.15),
                      FadeIn(parts[i], shift=RIGHT * 0.3), run_time=0.7)
        self.play(Write(plus))

        # output pixel (single channel out)
        out = grid(rows - K + 1, cols - K + 1, 0.5, color=GREEN_B)
        out.to_edge(RIGHT, buff=1.1).shift(DOWN * 0.2)
        outlab = mtext("output[0]", 0.36, GREEN_B).next_to(out, UP, buff=0.15)
        self.play(FadeIn(out), Write(outlab))

        tgt = out.cellat(pr, pc)
        arrows = VGroup(*[Arrow(parts[i].get_right(), tgt.get_left(), buff=0.15,
                                stroke_width=3, color=CH_COL[i],
                                max_tip_length_to_length_ratio=0.12) for i in range(3)])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.1),
                  tgt.animate.set_fill(GREEN_B, 0.95), run_time=1.0)

        eq = mtext("out[oc][y][x] = sum over ic,ky,kx  of  in[ic]*w[oc][ic]", 0.4, HL)
        eq.to_edge(DOWN, buff=0.4)
        self.play(Write(eq))
        self.wait(1.3)


# ===========================================================================
class ChannelParallel(Scene):
    """Cout filters each make an independent output plane -> embarrassingly parallel."""

    def construct(self):
        rows = cols = 5
        Cout = 4

        title = mtext("each output channel (Cout) is INDEPENDENT -> parallel", 0.46)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))

        # shared 3-channel input on the left
        planes, stack = channel_stack(3, rows, cols, 0.34, 0.2)
        stack.to_edge(LEFT, buff=0.7)
        inlab = mtext("input\n3 channels", 0.32, WHITE).next_to(stack, DOWN, buff=0.4)
        self.play(FadeIn(stack), Write(inlab))

        # Cout independent output planes on the right, stacked vertically
        outs = VGroup()
        for o in range(Cout):
            g = grid(rows - 2, cols - 2, 0.3, color=GREEN_B)
            outs.add(g)
        outs.arrange(DOWN, buff=0.3).to_edge(RIGHT, buff=1.6).shift(UP * 0.1)
        olabs = VGroup(*[mtext(f"out[{o}]", 0.28, GREEN_B).next_to(outs[o], LEFT, buff=0.2)
                         for o in range(Cout)])
        self.play(FadeIn(outs), *[Write(l) for l in olabs])

        # filter banks: each output channel has its own (Cin-deep) filter
        flabs = VGroup(*[mtext(f"W[{o}]", 0.26, YELLOW_E) for o in range(Cout)])
        for o in range(Cout):
            flabs[o].next_to(outs[o], UP, buff=0.04).shift(LEFT * 1.1)

        # PARALLEL: every output plane fills at the same time, each fed by the
        # full 3-channel input. lag_ratio=0 -> simultaneous (the whole point).
        feeds = VGroup()
        for o in range(Cout):
            a = Arrow(stack.get_right(), outs[o].get_left(), buff=0.2,
                      stroke_width=3, color=interpolate_color(BLUE, PURPLE, o / Cout),
                      max_tip_length_to_length_ratio=0.06)
            feeds.add(a)
        self.play(LaggedStart(*[GrowArrow(a) for a in feeds], lag_ratio=0.0,
                              run_time=0.8),
                  *[Write(flabs[o]) for o in range(Cout)])

        fill_anims = []
        for o in range(Cout):
            for cell in outs[o]:
                fill_anims.append(cell.animate.set_fill(GREEN_B, 0.9))
        self.play(*fill_anims, run_time=1.2)   # all planes at once

        # annotate the two axes
        box1 = mtext("Cout  ->  PARALLEL  (threads / cores, no sync)", 0.36, GREEN_B)
        box2 = mtext("Cin   ->  REDUCTION (accumulate, dependency chain)", 0.36, CH_COL[0])
        VGroup(box1, box2).arrange(DOWN, buff=0.18, aligned_edge=LEFT) \
            .to_edge(DOWN, buff=0.35)
        self.play(Write(box1))
        self.play(Write(box2))
        self.wait(1.4)
