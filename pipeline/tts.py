"""Pluggable on-device TTS. Two engines, same interface:

    synth(text, out_wav, engine=..., voice=...) -> duration_seconds

  engine="say"    : macOS built-in `say` (zero setup, always works, on-device)
  engine="kokoro" : mlx-audio Kokoro-82M in the isolated `mlx-tts` conda env
                    (neural, far nicer; needs the env + weights)

Both write a 24kHz mono wav so the downstream ffmpeg mux is uniform.
"""
import glob
import os
import re
import subprocess
import wave

MLX_TTS_PY = "/Users/briancpark/miniforge3/envs/mlx-tts/bin/python"
KOKORO_MODEL = "prince-canuma/Kokoro-82M"


def _dur(wav_path):
    with wave.open(wav_path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def _to_wav(src, dst):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-ar", "24000", "-ac", "1", dst], check=True)


def synth_say(text, out_wav, voice="Samantha", rate=170):
    aiff = out_wav + ".aiff"
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", aiff, text], check=True)
    _to_wav(aiff, out_wav)
    os.remove(aiff)
    return _dur(out_wav)


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _concat_wavs(segs, out_wav, workdir):
    if len(segs) == 1:
        _to_wav(segs[0], out_wav)
        return
    lst = os.path.join(workdir, "_concat_list.txt")
    with open(lst, "w") as f:
        for s in segs:
            f.write(f"file '{os.path.abspath(s)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", lst, "-ar", "24000", "-ac", "1",
                    out_wav], check=True)
    os.remove(lst)


def _kokoro_one(text, prefix, workdir, voice, speed):
    """Generate one short chunk with Kokoro; return its wav path (raises on failure)."""
    for stale in glob.glob(os.path.join(workdir, prefix + "*.wav")):
        os.remove(stale)
    subprocess.run(
        [MLX_TTS_PY, "-m", "mlx_audio.tts.generate", "--model", KOKORO_MODEL,
         "--voice", voice, "--speed", str(speed), "--text", text,
         "--file_prefix", prefix], check=True, cwd=workdir,
        capture_output=True, text=True)
    hits = sorted(glob.glob(os.path.join(workdir, prefix + "*.wav")))
    if not hits:
        raise RuntimeError(f"kokoro produced no wav for {prefix!r}")
    if len(hits) > 1:                      # rare internal split: merge them
        merged = os.path.join(workdir, prefix + "_merged.wav")
        _concat_wavs(hits, merged, workdir)
        return merged
    return hits[0]


def _kokoro_chunk(text, seg_out, tag, workdir, voice, speed):
    """Render one chunk to seg_out with Kokoro. On Kokoro's shape bug, split the
    chunk in half and recurse (shorter inputs succeed); only the smallest
    irreducible chunk that still fails falls back to `say`. Keeps one voice."""
    try:
        src = _kokoro_one(text, tag, workdir, voice, speed)
        _to_wav(src, seg_out)
        return
    except (subprocess.CalledProcessError, RuntimeError):
        pass
    words = text.split()
    if len(words) <= 3:                       # irreducible -> say
        print(f"    [kokoro can't handle {text!r}; using say]")
        synth_say(text, seg_out)
        return
    mid = len(words) // 2
    a = os.path.join(workdir, tag + "_a.wav")
    b = os.path.join(workdir, tag + "_b.wav")
    _kokoro_chunk(" ".join(words[:mid]), a, tag + "a", workdir, voice, speed)
    _kokoro_chunk(" ".join(words[mid:]), b, tag + "b", workdir, voice, speed)
    _concat_wavs([a, b], seg_out, workdir)
    for f in (a, b):
        if os.path.exists(f):
            os.remove(f)


def synth_kokoro(text, out_wav, voice="af_heart", speed=1.0):
    """Synthesize sentence-by-sentence (short inputs dodge Kokoro's shape bug),
    recursively splitting any sentence Kokoro can't handle. Robust + consistent."""
    if not os.path.exists(MLX_TTS_PY):
        raise RuntimeError("mlx-tts env not ready; use engine='say'")
    out_wav = os.path.abspath(out_wav)
    d = os.path.dirname(out_wav)
    base = os.path.basename(out_wav)[:-4]
    segs = []
    for i, sent in enumerate(_split_sentences(text)):
        seg_out = os.path.join(d, f"{base}_s{i:02d}.wav")
        _kokoro_chunk(sent, seg_out, f"{base}_k{i:02d}", d, voice, speed)
        segs.append(seg_out)
    _concat_wavs(segs, out_wav, d)
    for s in segs + glob.glob(os.path.join(d, base + "_k*.wav")):
        if os.path.exists(s) and s != out_wav:
            os.remove(s)
    return _dur(out_wav)


def synth(text, out_wav, engine="say", voice=None):
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    if engine == "kokoro":
        return synth_kokoro(text, out_wav, voice or "af_heart")
    return synth_say(text, out_wav, voice or "Samantha")


if __name__ == "__main__":
    import sys
    eng = sys.argv[1] if len(sys.argv) > 1 else "say"
    d = synth("Convolution, made fourteen times faster with NEON.",
              "/tmp/tts_probe.wav", engine=eng)
    print(f"{eng}: {d:.2f}s -> /tmp/tts_probe.wav")
