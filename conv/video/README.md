# conv2d video pipeline

Turns the Manim scenes into narrated videos with on-device TTS and background music.

## Outputs
- `out/short_final.mp4` — vertical 1080×1920 YouTube Short (~30s)
- `out/long_final.mp4`  — horizontal 1920×1080 explainer (the 6-scene walkthrough)

## Build
```bash
# from conv/
make short          # vertical short  (Kokoro voice)
make long           # long explainer  (Kokoro voice)
make short VOICE=say   # use macOS `say` instead (no env needed)
```
Or directly:
```bash
python3 video/build.py short kokoro
python3 video/build.py long  say
```

## TTS engines (on-device)
| engine | quality | setup |
|--------|---------|-------|
| `kokoro` | neural, YouTube-grade | isolated `mlx-tts` conda env + Kokoro-82M weights (already installed) |
| `say`    | basic, robust | none — macOS built-in |

Kokoro runs in its own env (`/Users/briancpark/miniforge3/envs/mlx-tts`) so it can't
disturb your `mlx`/`exo` setup. Voice is `af_heart`; change it in `tts.py`.
Edit narration text in `narration.py`.

## Music
Drop **one** audio file in `video/music/` (mp3/m4a/wav/flac). The builder loops it,
lowers it to ~16%, and **ducks it under the voice** via ffmpeg `sidechaincompress`.
With no file present, the video is built voice-only.

`music/ambient_pad.wav` is a synthesized placeholder (original, royalty-free).
Replace it with a real track to taste.

⚠️ **Copyright:** Do not use Mark Rober's (or anyone's) copyrighted music unless you
have a license — YouTube Content ID will claim it on upload. Royalty-free sources:
YouTube Audio Library, Kevin MacLeod / incompetech (CC-BY, credit required),
Pixabay Music, Free Music Archive. Check each track's license.

## How sync works
Each scene clip length = `max(animation, narration) + 0.4s`. If narration is longer
than the animation, the last frame freezes to fill. Keep narration in `narration.py`
close to each scene's animation length to minimize freezing.

## Extending toward a 10-minute cut
The current long video is ~5–6 min (6 scenes). To reach 10 min honestly (not by
freezing), add scenes + matching narration for: FMA latency vs throughput, cache
blocking/tiling, multithreading over output channels, and a roofline/GFLOP-s recap.
Add them to the relevant `*.py`, then list them in `LONG` in `narration.py`.
