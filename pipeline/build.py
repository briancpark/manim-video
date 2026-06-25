#!/usr/bin/env python3
"""Shared narrated-video builder for any topic in this repo.

  python pipeline/build.py <topic> short [say|kokoro]   -> vertical Short
  python pipeline/build.py <topic> long  [say|kokoro]   -> horizontal explainer

<topic> is a directory at the repo root (e.g. conv, gemm). It must contain:
  <topic>/video/narration.py   with LONG = [{scene:(module,Class), text:...}, ...]
                               and optionally SHORT = {scene:(module,Class), text:...}
  <topic>/<module>.py          the Manim scene files (rendered on demand)

Per scene: render (if needed) -> TTS -> fit clip to max(video,narration) by
freezing the last frame -> concat -> duck music under the voice -> mp4 in
<topic>/video/out/.

Background music: drop ONE audio file in <topic>/video/music/. None -> voice only.
Do NOT use copyrighted tracks you don't have a license for (Content-ID will
claim them). Royalty-free: YouTube Audio Library, incompetech, Pixabay Music.
"""
import glob
import importlib.util
import os
import subprocess
import sys

PIPE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(PIPE)
MANIM = "/Users/briancpark/miniforge3/envs/manim-ce/bin/manim"
QDIR = {"-ql": "480p15", "-qm": "720p30", "-qh": "1080p60", "-qk": "2160p60"}

sys.path.insert(0, PIPE)
from tts import synth


def sh(args, **kw):
    return subprocess.run(args, check=True, **kw)


def probe_dur(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    return float(out)


def load_narration(video_dir):
    spec = importlib.util.spec_from_file_location(
        "narration", os.path.join(video_dir, "narration.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Topic:
    def __init__(self, name):
        self.root = os.path.join(REPO, name)
        self.video = os.path.join(self.root, "video")
        self.out = os.path.join(self.video, "out")
        self.work = os.path.join(self.video, "work")
        self.music = os.path.join(self.video, "music")
        if not os.path.isdir(self.video):
            raise SystemExit(f"no {self.video} (need narration.py there)")
        self.N = load_narration(self.video)
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(self.work, exist_ok=True)

    def find_or_render(self, module, cls, quality="-qh"):
        qd = QDIR.get(quality, "1080p60")
        target = os.path.join(self.root, "media", "videos", module, qd, cls + ".mp4")
        if os.path.exists(target):
            return target
        print(f"  rendering {module}.{cls} @ {qd} ...")
        sh([MANIM, quality, "--format", "mp4",
            os.path.join(self.root, module + ".py"), cls], cwd=self.root)
        if not os.path.exists(target):
            raise RuntimeError(f"render produced no file at {target}")
        return target

    def find_music(self):
        for ext in ("mp3", "m4a", "wav", "flac", "aac", "ogg"):
            hits = glob.glob(os.path.join(self.music, f"*.{ext}"))
            if hits:
                return hits[0]
        return None


def scene_clip(video, wav, dst, fps=30, w=None, h=None):
    """Normalized clip: video frozen to max(video,narration), audio padded."""
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


def concat(clips, dst, workdir):
    lst = os.path.join(workdir, "concat.txt")
    with open(lst, "w") as f:
        for c in clips:
            f.write(f"file '{c}'\n")
    sh(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", lst, "-c", "copy", dst])


def add_music(video, music, dst, vol=0.16):
    total = probe_dur(video)
    fc = (f"[1:a]volume={vol}[m];"
          f"[m][0:a]sidechaincompress=threshold=0.02:ratio=6:attack=20:release=400[md];"
          f"[0:a][md]amix=inputs=2:duration=first:normalize=0[a]")
    sh(["ffmpeg", "-y", "-loglevel", "error", "-i", video,
        "-stream_loop", "-1", "-i", music, "-filter_complex", fc,
        "-map", "0:v", "-map", "[a]", "-t", f"{total:.3f}",
        "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "2", dst])


def finalize(body, t, name):
    music = t.find_music()
    final = os.path.join(t.out, name)
    if music:
        print(f"  music: {os.path.basename(music)} (ducked)")
        add_music(body, music, final)
    else:
        print("  no music found -> voice only")
        os.replace(body, final)
    print(f"DONE -> {final}  ({probe_dur(final):.1f}s)")


def build_short(t, engine):
    if not hasattr(t.N, "SHORT"):
        raise SystemExit("this topic has no SHORT in narration.py")
    mod, cls = t.N.SHORT["scene"]
    video = t.find_or_render(mod, cls)
    wav = os.path.join(t.work, "short_vo.wav")
    print(f"  TTS ({engine}) ...")
    synth(t.N.SHORT["text"], wav, engine=engine)
    clip = os.path.join(t.work, "short_clip.mp4")
    scene_clip(video, wav, clip, fps=30)
    finalize(clip, t, "short_final.mp4")


def build_long(t, engine):
    clips = []
    for i, beat in enumerate(t.N.LONG):
        mod, cls = beat["scene"]
        print(f"[{i+1}/{len(t.N.LONG)}] {cls}")
        video = t.find_or_render(mod, cls)
        wav = os.path.join(t.work, f"long_{i:02d}.wav")
        print(f"  TTS ({engine}) ...")
        synth(beat["text"], wav, engine=engine)
        clip = os.path.join(t.work, f"long_{i:02d}.mp4")
        scene_clip(video, wav, clip, fps=30, w=1920, h=1080)
        clips.append(clip)
    body = os.path.join(t.work, "long_body.mp4")
    concat(clips, body, t.work)
    finalize(body, t, "long_final.mp4")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: build.py <topic> <short|long> [say|kokoro]")
    topic = Topic(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "long"
    engine = sys.argv[3] if len(sys.argv) > 3 else "kokoro"
    (build_short if mode == "short" else build_long)(topic, engine)
