# 🎬 Director's Notes — *HPC, Visualized*

The single source of truth for the channel's voice, look, and sound.
If a creative decision isn't here, it should end up here.

---

## North Star

> **Fuse Mark Rober's engineering showmanship with 3Blue1Brown's visual rigor.**

Mark Rober makes you *feel* the payoff — build-up, suspense, then a satisfying
"WHOA." 3Blue1Brown makes you *understand* the mechanism — clean animation,
one idea at a time, no hand-waving. We want both in the same breath: a viewer
should learn *exactly how* a kernel gets fast **and** get goosebumps when it does.

**The thesis of every video:** *Same math. Feed the hardware better. Watch it fly.*

---

## Punchlines

Each video is built around one punchline — the sentence the whole thing exists to land.

| Video | Punchline |
|-------|-----------|
| **Series** | "It's the *same math* — the speed comes from respecting the hardware." |
| **conv2d** | "One weight, broadcast across four lanes, sixteen in flight — **14× faster, bit-for-bit identical.**" |
| **conv → im2col** | "Convolution *is* a matrix multiply — so it inherits decades of tuned BLAS for free." |
| **conv channels** | "**Reduce over input channels, parallelize over output channels.** That's the whole recipe." |
| **gemm** | "Cache blocking + a register microkernel = **40× over naive** — and that's before special hardware." |
| **gemm → Tensor Cores** | "CPUs hit a ceiling. Tensor Cores *raise the roof* — the same `D = A·B + C`, but in silicon built only for matrices." |

---

## Style guide

### Visual (the 3b1b half)
- **Engine:** Manim Community, dark background, monospace (`Menlo`) labels.
- **Palette:** input = blue (`BLUE_B`), weights/kernel = yellow (`YELLOW_E`),
  output = green (`GREEN_B`), highlight/action = orange (`ORANGE`),
  tensor-core/accelerator = violet (`#B67CFF`), memory = blue (`#4D8DFF`).
- **One idea per scene.** Build it up element by element; never dump a full diagram.
- **Color = meaning.** A color always maps to the same role across all videos.
- **Motion shows the mechanism:** sliding windows, broadcast splats, FMA arrows,
  data moving through the memory hierarchy. If it doesn't teach, cut it.

### Narration (the explainer half)
- **Voice:** Kokoro `af_heart` (neural, on-device). Warm, calm, confident.
- **Tone:** plain-spoken expert. Short declarative sentences. No jargon without
  an immediate plain-English gloss ("a reduction — a dependency chain that must
  accumulate").
- **Arc per video:** hook → naive baseline → "here's the trick" → mechanism →
  **payoff** → one-line takeaway.
- **Pacing:** ~150 wpm. Let a beat breathe before the reveal.

### Music (the Mark Rober half) — *the secret sauce*
Music is **scored to the narrative**, not wallpaper. The arc:

```
setup ........ chill lo-fi bed (curious, low)
build ........ warmer groove rises
REVEAL/aha ... energetic track swells UP under the punchline
finale ....... the Arrow theme (our signature) lands the takeaway
```

Rules:
- **Bed under explanation, swell at the "aha".** Volume tracks the emotional beat.
- **Ducking:** music sits under the voice via sidechain compression, then
  **rises in the gaps** (scene transitions, post-punchline holds) — that breath
  is where the Rober energy lives.
- **"Arrow" (Andrew Applepie — the actual Mark Rober theme) is our signature.**
  It plays the **finale scene** of every long video and carries the Shorts.
  Recurrence = branding.
- All tracks are no-copyright / royalty-free (Andrew Applepie, Blue Wednesday,
  Joakim Karud, etc.), stored in `music/` (git-ignored — not redistributed).

---

## Music library → mood

Probed by integrated loudness + feel. Lower LUFS ≈ calmer bed.

| Mood | Use | Tracks |
|------|-----|--------|
| **Bed** (calm, ~−26..−13 LUFS) | setup / explanation | `10 Cereal Killa`, `08 Dive`, `11 The Ocean`, `15 Almost Original`, `17 Falling`, `03 New Shoes` |
| **Build** (curious, playful) | new idea / momentum | `09 Pokemon in NYC`, `07 Berlin`, `14 spark`, `06 Dizzy`, `12 I'm So` |
| **Climax** (driving, ~−7 LUFS) | the reveal / "aha" | `02 Run (Part 2)`, `13 Keep On Trying`, `05 Dance`, `18 Too Happy To Be Cool`, `20 Dansez`, `19 Pata Pata` |
| **Signature** | finale / Shorts | `01 Arrow` (Mark Rober theme) |

---

## The scores

`intensity` sets pre-duck volume + swell. `bed`=0.11, `build`=0.16,
`climax`=0.24, `hero`=0.30 (hero/climax also fade *up* into the beat).
Lives machine-readable as `MUSIC`/`SHORT` in each topic's `video/narration.py`.

### conv2d — long (6 scenes)
| # | scene | beat | track | intensity |
|---|-------|------|-------|-----------|
| 1 | NaiveConv | slow baseline | `10 Cereal Killa` | bed |
| 2 | NeonConv | **the trick** | `13 Keep On Trying` | climax |
| 3 | Im2Col | new idea | `09 Pokemon in NYC` | build |
| 4 | Im2ColGEMM | **it's a matmul!** | `02 Run (Part 2)` | climax |
| 5 | ChannelReduce | multichannel | `08 Dive` | bed |
| 6 | ChannelParallel | **finale / recipe** | `01 Arrow` | hero |

### conv2d — Short (vertical, ~28s)
Single hero cut on `01 Arrow`, swell into the "14×" payoff.

### gemm — long (4 scenes)
| # | scene | beat | track | intensity |
|---|-------|------|-------|-----------|
| 1 | Tiling | cache setup | `03 New Shoes` | bed |
| 2 | Microkernel | **40× reveal** | `13 Keep On Trying` | climax |
| 3 | TensorCore | **the WOW (silicon)** | `02 Run (Part 2)` | hero |
| 4 | Roofline | **finale / synthesis** | `01 Arrow` | hero |

---

## Pipeline (how it's made)
- Scenes: `<topic>/*.py` (Manim CE in the `manim-ce` conda env).
- Narration + score: `<topic>/video/narration.py` (`LONG`, optional `SHORT`, `MUSIC`).
- Build: `pipeline/build.py <topic> <long|short> [kokoro|say]` →
  render → TTS → sync each clip to `max(anim, narration)` → **compose scored
  music** → duck under voice → `<topic>/video/out/*.mp4`.
- TTS: `pipeline/tts.py` (Kokoro neural / macOS `say`).

## Conventions
- Each topic = its own top-level dir, mirroring `conv/` and `gemm/`.
- Final renders live in `<topic>/video/out/`; intermediates in `work/` (ignored).
- Keep the README punchy; keep the *why* here.
