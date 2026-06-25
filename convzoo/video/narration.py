"""Script + music score for "The Convolution Zoo" — every convolution variant,
how each maps to GEMM, how that runs on NEON / Arm SME / NVIDIA Tensor Cores,
and why that's what powers GenAI. Built on director.md (completion-first:
cold-open payoff, open loops, setback-before-reveal, discovery tone).

Consumed by ../../pipeline/build.py. MUSIC is the per-scene score.
"""

# Per-scene music score, parallel to LONG (see director.md).
MUSIC = [
    {"track": "01", "intensity": "hero",   "offset": 0},   # ColdOpen      - Arrow (signature)
    {"track": "09", "intensity": "build",  "offset": 4},   # SpatialFamily - Pokemon in NYC
    {"track": "13", "intensity": "climax", "offset": 16},  # GemmUnify     - Keep On Trying
    {"track": "07", "intensity": "build",  "offset": 10},  # ChannelZoo    - Berlin
    {"track": "14", "intensity": "build",  "offset": 8},   # ShapeTricks   - spark
    {"track": "06", "intensity": "bed",    "offset": 0},   # NEON          - Dizzy
    {"track": "02", "intensity": "climax", "offset": 12},  # SME           - Run (Part 2)
    {"track": "05", "intensity": "climax", "offset": 8},   # TensorCore    - Dance
    {"track": "20", "intensity": "climax", "offset": 8},   # WhyMatrixWins - Dansez
    {"track": "01", "intensity": "hero",   "offset": 0},   # GenAI         - Arrow (finale)
]

LONG = [
    {   # 1 — COLD OPEN: payoff + open loop, then rewind
        "scene": ("convzoo_anim", "ColdOpen"),
        "text": (
            "Every image an AI dreams up, every video it generates, every voice it "
            "clones, is built on one operation you've probably never thought about: "
            "the convolution. But here's the strange part. There isn't one convolution, "
            "there's a whole zoo of them, one, two, and three dimensional, pointwise, "
            "depthwise, grouped, dilated, transposed. They look completely different. "
            "And yet, underneath, they are almost all the exact same piece of math. "
            "Let's see how the whole zoo collapses into one idea, and how that idea is "
            "what makes modern AI fast enough to exist."
        ),
    },
    {   # 2 — SPATIAL FAMILY 1D/2D/3D
        "scene": ("convzoo_anim", "SpatialFamily"),
        "text": (
            "Start with the family you already know. A convolution slides a small "
            "kernel across data and, at every position, takes a weighted sum. "
            "In one dimension, the kernel slides along a line, perfect for audio and "
            "time series. In two dimensions, it slides across a grid, the classic image "
            "filter. In three, it sweeps through a volume, across space and time, which "
            "is how AI understands video. Same operation, one more axis each time. "
            "The cost, though, explodes: a three by three by three kernel over a video "
            "is a lot of multiplies. So how does any of this run fast?"
        ),
    },
    {   # 3 — THE UNIFYING TRICK: im2col -> GEMM
        "scene": ("convzoo_anim", "GemmUnify"),
        "text": (
            "Here's the trick that runs the whole zoo. Take every patch the kernel "
            "lands on and flatten it into a column. Stack those columns into a matrix. "
            "Flatten the filters into rows. Now the entire convolution, all those "
            "sliding windows, is a single matrix multiply. This is called im to col, "
            "and it's the key that unlocks everything, because matrix multiply is the "
            "one operation computer hardware has been tuned to do faster than anything "
            "else. So the real question stops being how do we make convolution fast, "
            "and becomes how do we make matrix multiply fast. Hold that thought."
        ),
    },
    {   # 4 — CHANNEL ZOO: 1x1, grouped, depthwise, separable
        "scene": ("convzoo_anim", "ChannelZoo"),
        "text": (
            "Now the zoo gets interesting, by playing with channels. Shrink the kernel "
            "to one by one and it stops mixing space entirely; it only mixes channels, "
            "which is, quite literally, just a matrix multiply with no flattening at all. "
            "Split the channels into independent groups and you get grouped convolution, "
            "cheaper, and trivially parallel. Push that to the extreme, one group per "
            "channel, and you get depthwise convolution, where each channel is filtered "
            "alone. Depthwise is the rebel of the zoo: with nothing to contract across "
            "channels, it refuses to become one big matrix multiply. Pair it with a one "
            "by one, and you get the depthwise separable trick that made MobileNet run "
            "on your phone, nearly nine times fewer multiplies for almost the same result."
        ),
    },
    {   # 5 — SHAPE TRICKS: dilated + transposed
        "scene": ("convzoo_anim", "ShapeTricks"),
        "text": (
            "Two more shapes, two more powers. Dilated convolution spreads the kernel's "
            "taps apart, leaving gaps, so it sees far across the input without any extra "
            "weights, how WaveNet hears seconds of audio at once. And transposed "
            "convolution runs the whole thing in reverse: instead of shrinking the input, "
            "it grows it, spreading each pixel back out into a larger grid. That's "
            "learnable upsampling, how a generator paints a tiny latent back up into a "
            "full image. Different shapes, different jobs, but notice: every single one "
            "still lowers to that same matrix multiply."
        ),
    },
    {   # 6 — NEON (SIMD)
        "scene": ("convzoo_anim", "NEON"),
        "text": (
            "So everything is a matrix multiply. How does the silicon actually do it? "
            "Start with a CPU's vector unit, Arm NEON. It works on four numbers at once. "
            "Take one weight, broadcast it across four lanes, and fuse-multiply-add it "
            "against four inputs in a single instruction. Keep a tile of results live in "
            "registers and stream through. It's fast, but notice the limit: each number "
            "you load gets reused along just one axis. Load four, do four multiplies. "
            "The hardware is asking a question, can we reuse data more than that?"
        ),
    },
    {   # 7 — SME (Arm matrix engine)
        "scene": ("convzoo_anim", "SME"),
        "text": (
            "Arm's answer is the Scalable Matrix Extension, SME, now shipping in Apple's "
            "M4. Instead of a vector, it keeps a two dimensional accumulator on chip, "
            "called the ZA tile. Then, in one instruction, it takes a column and a row "
            "and computes their full outer product, multiplying every element of one "
            "against every element of the other, and adds the whole grid into the tile. "
            "One instruction, hundreds of multiply-adds. That's the leap: a vector loaded "
            "once is now reused across an entire dimension, not just one lane."
        ),
    },
    {   # 8 — TENSOR CORE
        "scene": ("convzoo_anim", "TensorCore"),
        "text": (
            "NVIDIA pushes the same idea even harder with Tensor Cores. Here a whole "
            "group of thirty-two threads, a warp, cooperates on one tile. Small fragments "
            "of A and B are staged down the memory hierarchy into registers, and then a "
            "single instruction computes D equals A times B plus C, an entire little "
            "matrix multiply, in one shot, in mixed precision: low precision multiply, "
            "high precision accumulate. Thousands of multiply-adds per instruction. This "
            "is the engine the entire AI industry now runs on."
        ),
    },
    {   # 9 — WHY MATRIX HARDWARE WINS
        "scene": ("convzoo_anim", "WhyMatrixWins"),
        "text": (
            "Here's why this matters so much. A vector unit reuses each loaded value "
            "along one axis, so the work grows with the length, L. A matrix engine "
            "computes an outer product, so loading the same two vectors produces L "
            "squared multiply-adds, the entire grid. Linear versus quadratic, from the "
            "same memory traffic. On the roofline, that's the move from starved and "
            "memory-bound, up and to the right, until you hit the compute ceiling, and "
            "then dedicated matrix hardware raises the ceiling itself. Same math. Feed "
            "the hardware better, and use the right units."
        ),
    },
    {   # 10 — GENAI PAYOFF + button
        "scene": ("convzoo_anim", "GenAI"),
        "text": (
            "And now the payoff. Where does the whole zoo actually live today? The image "
            "models, Stable Diffusion's U-Net and its V-A-E, are built from three by "
            "three and one by one convolutions. The newer transformer models still begin "
            "by patchifying the image, which is just a strided convolution. Video models "
            "compress time with causal three dimensional convolutions. Audio codecs and "
            "vocoders are towers of one dimensional convolutions. Even language models "
            "like Mamba carry a small depthwise conv inside every block. And every one "
            "of them, every variant in the zoo, is lowered to a matrix multiply and "
            "thrown at Tensor Cores. The same math, in a dozen disguises, feeding the "
            "engines that power generative AI. That's the convolution zoo."
        ),
    },
]
