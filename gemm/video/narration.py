"""Narration for the GEMM explainer: cache blocking -> NEON microkernel ->
NVIDIA tensor cores -> roofline. Consumed by ../../pipeline/build.py.

MUSIC is the per-scene score (see director.md). track = number in music/,
intensity in {bed, build, climax, hero}, offset = seconds into the track."""

# Parallel to LONG, scene-for-scene.
MUSIC = [
    {"track": "03", "intensity": "bed",    "offset": 0},   # Tiling      - New Shoes
    {"track": "13", "intensity": "climax", "offset": 16},  # Microkernel - Keep On Trying
    {"track": "02", "intensity": "hero",   "offset": 12},  # TensorCore  - Run (Part 2)
    {"track": "01", "intensity": "hero",   "offset": 0},   # Roofline    - Arrow (theme)
]

LONG = [
    {
        "scene": ("gemm_anim", "Tiling"),
        "text": (
            "A matrix multiply does an enormous amount of arithmetic, but its "
            "speed is usually limited by memory, not math. In the naive version, "
            "computing each row of the output reads the entire B matrix from DRAM. "
            "Do that for every row, and you stream B from memory again and again, "
            "so the processor spends most of its time just waiting. Cache blocking "
            "fixes this. We break the matrices into small tiles, load one block of B "
            "into the fast L1 cache, and reuse it for many output values before "
            "moving on. The data travels from slow memory once, then stays close to "
            "the cores. Same multiply, far less memory traffic."
        ),
    },
    {
        "scene": ("gemm_anim", "Microkernel"),
        "text": (
            "Inside each tile we hand-write a NEON microkernel. We keep an eight by "
            "eight block of the output live entirely in vector registers. Then for "
            "each step along the shared dimension, we take a column of A and a row of "
            "B and apply a rank-one update: an outer product that touches all "
            "sixty-four accumulators at once. Each B vector we load is reused by all "
            "eight rows, and each A value is reused across all eight columns. Nothing "
            "spills back to memory until the tile is finished. The result is a kernel "
            "limited only by how fast the fused multiply-add units can run, about "
            "forty times faster than the naive loop."
        ),
    },
    {
        "scene": ("gemm_anim", "TensorCore"),
        "text": (
            "CPUs eventually hit a ceiling. GPUs go further with dedicated matrix "
            "hardware called tensor cores, which push blocking to an extreme. Data is "
            "staged down the memory hierarchy: from high-bandwidth memory, into "
            "shared memory, and finally into small per-thread fragments held by a "
            "warp. Then, instead of scalar loops, a single warp-level instruction "
            "feeds the A and B fragments into the tensor core, which computes D "
            "equals A times B plus C as one fused matrix operation. Dozens of "
            "multiply-adds happen inside the unit every cycle, often in mixed "
            "precision. It is the same math, but executed by silicon built only for "
            "matrices."
        ),
    },
    {
        "scene": ("gemm_anim", "Roofline"),
        "text": (
            "The roofline ties it together. The diagonal is the memory bandwidth "
            "limit, and the flat lines are the compute ceilings. The naive kernel "
            "sits low and to the left, starved for data. Cache blocking and the NEON "
            "microkernel raise the arithmetic intensity, moving us right until we "
            "bump against the SIMD roof. Tensor cores raise the roof itself, to a "
            "level NEON simply cannot reach. The whole lesson of high-performance "
            "computing is in one picture: keep the same math, feed the hardware "
            "better, and use the right units, and you climb from under two gigaflops "
            "to thousands."
        ),
    },
]
