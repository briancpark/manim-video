# 🎬 Director's Notes — *HPC, Visualized*

A ground-up blueprint for making engineering explainers people actually finish.
Built from how Mark Rober engineers engagement, fused with 3Blue1Brown's visual
rigor. **This drives the script first — animation, voice, and music serve it.**

---

## 0. The one rule (everything serves this)

> **"Nobody shares a video they haven't finished watching. That's the trick."**
> — Mark Rober

We optimize for **completion**, not impressions. Every scene must earn the next
30 seconds of attention. If a beat doesn't create a reason to keep watching,
cut it.

---

## 1. The engagement engine (how Rober actually does it)

Each principle below = what Rober does → **how we do it for HPC**.

### 1.1 Feeling over facts
Rober: *"Create a visceral response — they have to feel amazed, shocked, happy."*
*"I can't teach you if I can't get your attention."*
→ **We make speed physical.** A naive loop should feel *painfully slow* on
screen; the optimized one should feel *fast*. Use a live speedometer, a ticking
GFLOP/s counter, a side-by-side race, a bar that shoots across the screen. The
viewer should feel the 14× before they understand it.

### 1.2 The hook formula (first ~15 seconds, 3 moves)
1. **Confirm the promise in 3 seconds** — show the payoff/number the title sold.
2. **Open a loop** — pose the question the whole video answers.
3. **Delay the payoff** — promise it's coming, then pull back to the start.
→ **Cold open on the win, then rewind.** e.g. *"This convolution runs 14× faster
than the textbook version — exact same answer. Here's the one idea that does it…
but first, why is the obvious way so slow?"*

### 1.3 Lead with the consequence, not the setup
Rober opens mid-action (car already speeding at the wall), stakes first.
→ **No throat-clearing.** Never open with "In this video we'll discuss…" Open
on the stakes: *"This loop runs billions of times inside every neural network.
Make it 14× faster and you've just saved a data center."*

### 1.4 Open loops + re-hooks (curiosity gaps)
Rober never resolves everything at once; he layers questions and re-hooks every
segment to reset attention.
→ **Re-hook every scene (~30–45s).** End each beat by teasing the next: *"That's
12× — but we're leaving half the chip idle. Watch this."* One open loop should
always be running.

### 1.5 Thriller pacing: tension ↔ relief
*"Mark paces his videos like thrillers — curiosity, setbacks, reveals."*
→ **Plant a setback before each reveal.** Don't go straight to the fast version.
Show the naive attempt, let it disappoint, *then* drop the trick. Alternate
build-up and release; the reveal only lands if tension preceded it.

### 1.6 Three-act spine
Hook (stakes) → Exploration (obstacles, false starts) → Payoff (satisfying
resolution + one-line takeaway).
→ Every topic gets this arc (template in §2).

### 1.7 Explore, don't lecture (vicarious learning)
Rober *learns with* the audience instead of teaching down. Failure is reframed
as the game — the **Super Mario Effect**: focus on the princess, not the pits.
→ **Narrate as discovery.** "What if we…? — no, that stalls. But if we keep four
in flight… *there* it is." Frame each optimization as solving a puzzle on screen,
not reciting a result. Dead ends are content, not embarrassments.

### 1.8 Ruthless condensation
Rober compressed 500 hours of squirrel footage into 15 minutes. Every second
fights for its place.
→ **Cut every frame that doesn't hook, teach, or pay off.** If a scene explains
without advancing tension or understanding, it's filler. Kill it.

### 1.9 Scale formats, not videos
Recognizable structure + new content each time (Glitter Bomb 1→6).
→ **"X, made N× faster" is our format.** Same spine (hook → naive → trick →
hardware → roofline payoff), new kernel each episode. Recurring signature beats
(the speedometer, the Arrow theme finale) = branding.

---

## 2. The HPC explainer template (concrete)

### Long video (~4–7 min)
| beat | time | job | feeling |
|------|------|-----|---------|
| **Cold open** | 0:00–0:05 | show the payoff number + the promise | *whoa* |
| **The loop** | 0:05–0:20 | open the question; state the stakes; rewind | curiosity |
| **Act 1 — the pit** | 0:20–1:30 | the naive way; make it *feel* slow; benchmark it | mild frustration |
| **re-hook** | — | "there's one idea that fixes this" | anticipation |
| **Act 2 — the climb** | 1:30–4:00 | the trick(s), each with a small setback then reveal; counter ticks up each time | rising excitement |
| **The hardware turn** | — | "CPUs hit a ceiling… now watch what dedicated silicon does" | escalation |
| **Act 3 — the payoff** | last ~60s | the roofline / final number; the *same math* punchline | satisfaction |
| **The button** | last ~8s | one-line takeaway + tease next episode | closure + new loop |

### Short (~30–45s, vertical)
Cold open on the number → one idea, one visual → swell → payoff number → tag.
No Act 1 slog; it's *all* hook and payoff.

---

## 3. The "feeling" toolkit (make code visceral)
- **The counter**: GFLOP/s or ms ticking live during a run.
- **The race**: naive vs optimized bars/dots moving in real time, side by side.
- **The speedometer/gauge** for throughput; needle pinning at the roof.
- **Scale anchors**: "this runs 10 billion times per inference" / "= a data center."
- **The reveal cut**: hold on the slow result a beat too long, then *snap* to fast.
- **Number morph**: `1.6 → 64 GFLOP/s` animating up on the payoff.
- **The roofline as a mountain**: you're climbing toward a ceiling; hardware raises it.

---

## 4. Punchlines (the sentence each video lands)
| video | punchline |
|-------|-----------|
| **Series** | "It's the *same math* — the speed comes from respecting the hardware." |
| **conv2d** | "One weight, four lanes, sixteen in flight — **14× faster, bit-for-bit identical.**" |
| **conv → im2col** | "Convolution *is* a matrix multiply — so it inherits decades of tuned BLAS for free." |
| **gemm** | "Cache blocking + a register microkernel = **40× over naive** — before special hardware." |
| **gemm → Tensor Cores** | "CPUs hit a ceiling. Tensor Cores *raise the roof* — same `D=A·B+C`, in silicon built for matrices." |

---

## 5. Style — in service of the engine

### Visual (the 3b1b half)
Manim CE, dark bg, `Menlo`. **Color = role, always:** input `BLUE_B`, weight
`YELLOW_E`, output `GREEN_B`, action/highlight `ORANGE`, accelerator `#B67CFF`,
memory `#4D8DFF`. One idea per scene, built up element by element. Motion must
teach. Add the §3 feeling devices — current scenes are too "calm 3b1b," not
enough "Rober payoff."

### Voice (Kokoro `af_heart`, neural — never `say` except as silent fallback)
Warm, curious, plain-spoken. Short declarative sentences. Narrate as discovery
(§1.7). Land the reveal, then *pause* before the next loop.

### Music — scored to the emotional curve (not wallpaper)
The curve from §2 IS the music plan: bed under Act 1, **swell into each reveal**,
energy through Act 2, the **Arrow theme (the real Mark Rober track) as the
finale signature**. Ducked under VO, rising in the gaps — that breath is where
the energy lives. Intensities: `bed .11 / build .16 / climax .24 / hero .30`,
climax/hero fade *up* into the beat. Library mood map + per-scene score table
maintained in each topic's `video/narration.py` (`MUSIC`).

---

## 6. Pre-publish checklist (every video)
- [ ] Payoff visible in first 5 seconds?
- [ ] An open loop running at every moment until the end?
- [ ] A setback before each reveal (tension → relief)?
- [ ] At least one *visceral* device (counter/race/gauge) on the main payoff?
- [ ] Re-hook every 30–45s?
- [ ] Every scene hooks, teaches, OR pays off — no pure filler?
- [ ] One-line takeaway + tease for next at the end?
- [ ] Music swells on the reveals, beds under explanation?

---

## 7. Pipeline & conventions
- Scenes: `<topic>/*.py` (Manim CE, `manim-ce` env).
- Script + score: `<topic>/video/narration.py` (`LONG`, optional `SHORT`, `MUSIC`).
- Build: `pipeline/build.py <topic> <long|short> [kokoro|say]` → render → TTS →
  sync each clip to `max(anim, narration)` → compose scored music → duck → out.
- Each topic = its own top-level dir (mirrors `conv/`, `gemm/`). Finals in
  `<topic>/video/out/`. Music library in `music/` (git-ignored, not redistributed).

---

## Sources (research behind §1)
- Mark Rober, *The Super Mario Effect* — TEDxPenn
- Samir Chaudry, *How Mark Rober Engineered the Perfect YouTube Strategy* (LinkedIn) & The Colin and Samir Show
- OpusClip, *Mark Rober's Growth Playbook*
- *How to Make Viral Videos (According to Mark Rober)* — Screenwriting from Iowa
- Stanford Daily, *Mark Rober on scientific storytelling* (2025)
