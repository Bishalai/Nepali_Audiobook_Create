"""Create an MP4 video: Devanagari text on screen, synced to the TTS audio.

Usage:
    python scripts\\make_video.py books\\sample_ne.txt
    python scripts\\make_video.py mybook.txt --out output --sentence
    python scripts\\make_video.py mybook.txt --font "Nirmala UI" --resolution 1920x1080

Flow:
  1. Ensures the speech exists: if the .txt is new/edited (detected via the
     manifest's source hash) or audio is missing/partial, it synthesizes first
     (same pipeline as `python -m tts text`, pause/resume compatible).
  2. Builds a timeline from the manifest - the exact chunk durations + silence
     gaps used to assemble the audio, so text and speech stay in sync.
  3. Burns the text onto a solid-color background with ffmpeg/libass (proper
     HarfBuzz shaping, so Devanagari conjuncts/matras render correctly) and
     muxes the audio -> <out>/video/<name>.mp4

Options:
  --sentence           show one sentence at a time (timings estimated
                       proportionally within each chunk) instead of the whole
                       paragraph/chunk
  --font NAME          font family (default "Nirmala UI", ships with Windows)
  --fontsize N         fixed font size (default: auto-fit per chunk)
  --resolution WxH     default 1280x720
  --bg COLOR           background color, default 0x141422
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tts.pipeline import (MANIFEST_VERSION, load_text_paragraphs,  # noqa: E402
                          source_hash, synthesize_paragraphs)
from tts.prep import split_sentences  # noqa: E402

DEFAULT_FONT = "Nirmala UI"          # ships with Windows 10/11, full Devanagari
DEFAULT_RES = (1280, 720)


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def ensure_audio(txt: Path, out_dir: Path, args) -> tuple[Path, dict]:
    """Return (audio file, manifest), synthesizing first if needed."""
    paragraphs = load_text_paragraphs(txt)
    if not paragraphs:
        sys.exit(f"error: {txt} contains no text")

    tag = txt.stem
    audio_dir = out_dir / "audio"
    manifest_path = audio_dir / tag / "manifest.json"
    audio = audio_dir / f"{tag}.mp3"
    if not audio.exists():
        audio = audio_dir / f"{tag}.wav"

    need = True
    manifest = None
    if manifest_path.exists() and audio.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest = None
        if (manifest and manifest.get("v") == MANIFEST_VERSION
                and manifest.get("source") == source_hash(paragraphs)
                and manifest.get("chunks")
                and all(c.get("status") == "done"
                        for c in manifest["chunks"].values())
                and audio.stat().st_mtime >= manifest_path.stat().st_mtime):
            need = False

    if need:
        print(f"synthesizing speech for {txt.name} first "
              f"(new, edited, or incomplete audio)...")
        from tts.engine import VitsEngine
        engine = VitsEngine(Path(args.vits_dir), Path(args.checkpoint),
                            device=args.device,
                            length_scale=args.length_scale)
        try:
            res = synthesize_paragraphs(
                tag, txt.name, paragraphs, engine, out_dir,
                max_chars=args.max_chars, gap_sentence=args.gap_sentence,
                gap_para=args.gap_para, gap_heading=args.gap_heading,
                pause_flag=None, deadline=None, no_resume=False)
        finally:
            engine.free()
        if res.paused or not res.output:
            sys.exit("error: synthesis incomplete - re-run "
                     f"`python -m tts text {txt}` to finish, then retry")
        audio = Path(res.output)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        print(f"reusing existing audio: {audio}")
    return audio, manifest


# --------------------------------------------------------------------------- #
# timeline + subtitles
# --------------------------------------------------------------------------- #
def build_timeline(manifest: dict, sentence_mode: bool) -> tuple[list, float]:
    """[(start, end, text)] - mirrors tts.pipeline.assemble() exactly."""
    chunks = sorted(manifest["chunks"].items(), key=lambda kv: int(kv[0]))
    events: list[tuple[float, float, str]] = []
    t = 0.0
    first = True
    for _, c in chunks:
        if c.get("status") != "done" or not c.get("text"):
            continue                       # skipped by assemble() too
        gap = float(c.get("gap", 0.0))
        if not first and gap > 0:
            t += gap
        dur = float(c.get("audio_seconds", 0.0))
        start, end = t, t + dur
        text = c["text"].strip()
        if sentence_mode:
            sents = [s for s in split_sentences(text) if s.strip()]
            if len(sents) > 1:
                total = sum(len(s) for s in sents)
                acc = 0
                for s in sents:
                    acc += len(s)
                    s_end = start + dur * acc / total
                    events.append((start, s_end, s.strip()))
                    start = s_end
            elif text:
                events.append((start, end, text))
        elif text:
            events.append((start, end, text))
        t = end
        first = False
    return events, t


def pick_fontsize(text: str, w: int, h: int, forced: int = 0,
                  min_fs: int = 24, max_fs: int = 56,
                  max_lines: int = 9) -> int:
    if forced:
        return forced
    for fs in range(max_fs, min_fs - 1, -2):
        chars_per_line = max(8, int((w - 140) / (fs * 0.56)))
        lines = max(1, math.ceil(len(text) / chars_per_line))
        if lines <= max_lines and lines * fs * 1.30 <= h - 90:
            return fs
    return min_fs


def _ts(t: float) -> str:
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def write_ass(events: list, path: Path, w: int, h: int, font: str,
              fontsize: int, fade_ms: int = 300) -> None:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Text,{font},{fontsize},&H00FFFFFF,&H000000FF,&H00181818,&H96000000,0,0,0,0,100,100,0,0,1,3,1,5,70,70,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for start, end, text in events:
        if end - start < 0.05:
            continue
        text = text.replace("{", "(").replace("}", ")").replace("\n", " ")
        fs = pick_fontsize(text, w, h, forced=fontsize)
        eff = f"{{\\fad({fade_ms},{fade_ms})\\fs{fs}}}"
        lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Text,,0,0,0,,"
                     f"{eff}{text}")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# video
# --------------------------------------------------------------------------- #
def render_mp4(audio: Path, ass: Path, out_mp4: Path, w: int, h: int,
               bg: str, workdir: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("error: ffmpeg not found on PATH")
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c={bg}:s={w}x{h}:r=25",
        "-i", str(audio.resolve()),
        "-vf", f"ass={ass.name}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        str(out_mp4.resolve()),
    ]
    subprocess.run(cmd, check=True, cwd=str(workdir))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="make_video", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("txt", help="UTF-8 Devanagari text file")
    ap.add_argument("--out", default="output",
                    help="same --out dir used for `tts text` (default: output)")
    ap.add_argument("--font", default=DEFAULT_FONT,
                    help=f'font family (default "{DEFAULT_FONT}")')
    ap.add_argument("--fontsize", type=int, default=0,
                    help="fixed font size (default: auto-fit)")
    ap.add_argument("--resolution", default="1280x720",
                    help="WxH (default 1280x720)")
    ap.add_argument("--bg", default="0x141422",
                    help="background color (default 0x141422)")
    ap.add_argument("--sentence", action="store_true",
                    help="show one sentence at a time instead of the "
                         "whole chunk")
    ap.add_argument("--fade-ms", type=int, default=300,
                    help="text fade in/out in ms (default 300)")
    # synthesis passthrough (used only if speech must be (re)generated)
    ap.add_argument("--checkpoint", default="models/vits/G_100000.pth")
    ap.add_argument("--vits-dir", default="third_party/vits")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-chars", type=int, default=180)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--gap-sentence", type=float, default=0.25)
    ap.add_argument("--gap-para", type=float, default=0.65)
    ap.add_argument("--gap-heading", type=float, default=1.0)
    args = ap.parse_args(argv)

    txt = Path(args.txt)
    if not txt.exists():
        sys.exit(f"error: {txt} not found")
    try:
        w, h = (int(x) for x in args.resolution.lower().split("x"))
    except ValueError:
        sys.exit("error: --resolution must look like 1280x720")

    out_dir = Path(args.out)
    audio, manifest = ensure_audio(txt, out_dir, args)

    events, total = build_timeline(manifest, args.sentence)
    if not events or total <= 0:
        sys.exit("error: no synthesized chunks found in the manifest")
    missing = sum(1 for c in manifest["chunks"].values()
                  if c.get("status") != "done")
    if missing:
        print(f"warning: {missing} chunk(s) had no audio and are missing "
              f"from the video")

    video_dir = out_dir / "video"
    workdir = video_dir / txt.stem
    workdir.mkdir(parents=True, exist_ok=True)
    ass = workdir / "subs.ass"
    write_ass(events, ass, w, h, args.font, args.fontsize, args.fade_ms)

    out_mp4 = video_dir / f"{txt.stem}.mp4"
    print(f"rendering {len(events)} text events over {total / 60:.1f} min "
          f"-> {out_mp4} ...")
    render_mp4(audio, ass, out_mp4, w, h, args.bg, workdir)

    size = out_mp4.stat().st_size / 1e6
    print("=" * 74)
    print(f"video             : {out_mp4}  ({size:.1f} MB)")
    print(f"duration          : {total / 60:.1f} min   "
          f"resolution: {w}x{h}   font: {args.font}")
    print(f"audio             : {audio}")
    print(f"subtitles (ASS)   : {ass}")
    print("=" * 74)


if __name__ == "__main__":
    main()
