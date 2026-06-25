# hpc — HPC visualized

A test bed for high-performance computing topics, where each topic ships:
1. **Fast C/C++** (NEON SIMD, etc.) with a benchmark + correctness harness,
2. **Manim animations** explaining the algorithm and the optimization, and
3. A **narrated video pipeline** (on-device TTS + ducked music) that turns the
   animations into a YouTube Short and a long-form explainer.

## Topics

### `conv/` — conv2d, 14× faster with NEON SIMD
Multi-channel 2D convolution. Naive scalar reference vs a NEON path that
vectorizes across output width (broadcast-weight FMAs, 16 pixels in flight).
Bit-exact, ~14× faster on an Apple M1 Pro.

```bash
cd conv
make && make run        # build + benchmark (naive vs NEON)
make anim               # render the algorithm animations
make anim-im2col        # im2col -> GEMM lowering
make anim-channels      # multi-channel: Cin reduction vs Cout parallelism
make short              # vertical YouTube Short (narrated)
make long               # long-form explainer (narrated)
```

See [`conv/video/README.md`](conv/video/README.md) for the video pipeline
(TTS engines, music, sync).

### `gemm/` — single-precision matrix multiply, 40× with NEON
`C = A*B` from a naive triple loop → cache-blocked (tiling + ikj order) → a
hand-written NEON 8×8 register microkernel. Bit-exact; ~40× over naive at 1024³
(~64 GFLOP/s single core), benchmarked against Apple's Accelerate BLAS (which
uses the AMX matrix coprocessor — a different league).

```bash
cd gemm
make && make run        # build + benchmark (naive / blocked / neon / BLAS)
./gemm 2048 2048 2048    # custom sizes
make anim               # Tiling, Microkernel, TensorCore, Roofline animations
make long               # narrated explainer (incl. NVIDIA Tensor Cores)
```

### `convzoo/` — The Convolution Zoo (flagship explainer)
Every convolution variant (1D/2D/3D, pointwise, grouped, depthwise, separable,
dilated, transposed), how they all collapse to **GEMM**, how that runs on
**NEON / Arm SME / NVIDIA Tensor Cores**, and why that's what powers GenAI
(diffusion U-Nets & VAEs, DiT patchify, video 3D convs, audio 1D convs, Mamba).
Built on `director.md` (cold-open payoff, open loops, setback-before-reveal).

```bash
cd convzoo
make anim               # render all 10 scenes
make long               # narrated explainer (~8–10 min)
```

## Shared video pipeline (`pipeline/`)
`pipeline/build.py <topic> <short|long> [kokoro|say]` turns any topic's Manim
scenes into a narrated video: render → on-device TTS → sync each clip → concat →
duck music under the voice. Each topic supplies `<topic>/video/narration.py`
(a `LONG` list, optional `SHORT`) and an optional track in `<topic>/video/music/`.
`pipeline/tts.py` provides Kokoro (neural, isolated `mlx-tts` env) and macOS `say`.

## Requirements
- Apple Silicon + clang (for the NEON C code)
- ffmpeg
- conda envs: `manim-ce` (Manim Community) for rendering, `mlx-tts`
  (mlx-audio + Kokoro) for neural TTS. macOS `say` works with no setup.

## License
MIT
