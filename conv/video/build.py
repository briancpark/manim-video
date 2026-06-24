#!/usr/bin/env python3
"""Assemble narrated videos from the Manim scenes.

  python build.py short [say|kokoro]     -> vertical Short with VO (+music)
  python build.py long  [say|kokoro]     -> horizontal explainer with VO (+music)

Pipeline per scene: render (if needed) -> TTS narration -> fit clip to
max(video, narration) by freezing the last frame -> concat -> duck music
under the voice -> final mp4 in video/out/.

Background music: drop ONE audio file in video/music/ (mp3/m4a/wav). If none is
present the video is built voice-only. Do NOT use copyrighted tracks you don't
have a license for (e.g. ripped YouTube music) -- it will be Content-ID claimed.
Royalty-free sources: YouTube Audio Library, Kevin MacLeod / incompetech,
Pixabay Music, Free Music Archive (check each track's license).
"""
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # conv/
OUT = os.path.join(HERE, "out")
WORK = os.path.join(HERE, "work")
MUSIC_DIR = os.path.join(HERE, "music")
MANIM = "/Users/briancpark/miniforge3/envs/manim-ce/bin/manim"

sys.path.insert(0, HERE)
import narration as N
from tts import synth


def sh(args, **kw):
    return subprocess.run(args, check=True, **kw)


def probe_dur(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    return float(out)


QDIR = {"-ql": "480p15", "-qm": "720p30", "-qh": "1080p60", "-qk": "2160p60"}


def find_or_render(module, cls, quality):
    """Return path to the rendered scene mp4 at the requested quality.

    Renders if that resolution isn't present yet (a lower-res render from an
    earlier run won't be reused -- we want the long video crisp)."""
    qd = QDIR.get(quality, "1080p60")
    target = os.path.join(ROOT, "media", "videos", module, qd, cls + ".mp4")
    if os.path.exists(target):
        return target
    print(f"  rendering {module}.{cls} @ {qd} ...")
    sh([MANIM, quality, "--format", "mp4", os.path.join(ROOT, module + ".py"), cls],
       cwd=ROOT)
    if not os.path.exists(target):
        raise RuntimeError(f"render produced no file at {target}")
    return target


def scene_clip(video, wav, dst, fps=30, w=None, h=None):
    """One normalized clip: video frozen to max(video,narration), audio padded."""
    vdur, adur = probe_dur(video), probe_dur(wav)
    T = max(vdur, adur) + 0.4
    pad = max(0.0, T - vdur)
    scale = f"scale={w}:{h}," if w else ""
    vf = f"[0:v]{scale}tpad=stop_mode=clone:stop_duration={pad:.3f},fps={fps}[v]"
    af = f"[1:a]apad=whole_dur={T:.3f}[a]"
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-i", wav,
        "-filter_complex", f"{vf};{af}", "-map", "[v]", "-map", "[a]",
        "-t", f"{T:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", dst])
    return T


def concat(clips, dst):
    lst = os.path.join(WORK, "concat.txt")
    with open(lst, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", lst, "-c", "copy", dst])


def find_music():
    for ext in ("mp3", "m4a", "wav", "flac", "aac", "ogg"):
        hits = glob.glob(os.path.join(MUSIC_DIR, f"*.{ext}"))
        if hits:
            return hits[0]
    return None


def add_music(video, music, dst, vol=0.16):
    """Loop music, duck it under the voice (sidechain), mix."""
    total = probe_dur(video)
    fc = (
        f"[1:a]volume={vol}[m];"
        f"[m][0:a]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=400[md];"
        f"[0:a][md]amix=inputs=2:duration=first:normalize=0[a]"
    )
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
        "-stream_loop", "-1", "-i", music, "-filter_complex", fc,
        "-map", "0:v", "-map", "[a]", "-t", f"{total:.3f}",
        "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "2", dst])


def build_short(engine):
    os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    mod, cls = N.SHORT["scene"]
    video = find_or_render(mod, cls, "-qh")
    wav = os.path.join(WORK, "short_vo.wav")
    print(f"  TTS ({engine}) ...")
    synth(N.SHORT["text"], wav, engine=engine)
    clip = os.path.join(WORK, "short_clip.mp4")
    scene_clip(video, wav, clip, fps=30)
    music = find_music()
    final = os.path.join(OUT, "short_final.mp4")
    if music:
        print(f"  music: {os.path.basename(music)} (ducked)")
        add_music(clip, music, final)
    else:
        print("  no music found -> voice only")
        os.replace(clip, final)
    print(f"DONE -> {final}  ({probe_dur(final):.1f}s)")


def build_long(engine):
    os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
    clips = []
    for i, beat in enumerate(N.LONG):
        mod, cls = beat["scene"]
        print(f"[{i+1}/{len(N.LONG)}] {cls}")
        video = find_or_render(mod, cls, "-qh")
        wav = os.path.join(WORK, f"long_{i:02d}.wav")
        print(f"  TTS ({engine}) ...")
        synth(beat["text"], wav, engine=engine)
        clip = os.path.join(WORK, f"long_{i:02d}.mp4")
        scene_clip(video, wav, clip, fps=30, w=1920, h=1080)
        clips.append(clip)
    body = os.path.join(WORK, "long_body.mp4")
    concat(clips, body)
    music = find_music()
    final = os.path.join(OUT, "long_final.mp4")
    if music:
        print(f"  music: {os.path.basename(music)} (ducked)")
        add_music(body, music, final)
    else:
        print("  no music found -> voice only")
        os.replace(body, final)
    print(f"DONE -> {final}  ({probe_dur(final):.1f}s)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "short"
    engine = sys.argv[2] if len(sys.argv) > 2 else "say"
    (build_short if mode == "short" else build_long)(engine)
