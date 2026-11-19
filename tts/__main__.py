"""CLI: python -m tts <command>

Commands:
  list   <epub>                       list chapters (shows which are Nepali)
  run    <epub> --chapters 4          synthesize speech for translated chapters
  run    <epub> --all                 whole book
  pause                                ask a running `run` to pause gracefully

Pause / resume:
  * `python -m tts pause`             writes a PAUSE flag the running job checks
                                       after every sentence chunk
  * Ctrl+C                            also pauses safely
  * `--pause-after-min N`             automatic pause after N minutes
  Every pause writes the partial chapter audio built from the chunks done so
  far; re-run the same `run` command to resume from the manifest checkpoints.

Notes:
  * Input is the TRANSLATED (Devanagari) EPUB from `python -m translator run`.
  * Numbers are spelled out in Nepali words automatically (110 -> एक सय दस)
    because the VITS model cannot read digits.
  * Text is synthesized in sentence-sized chunks (default <= 180 chars) -
    long inputs degrade speech quality.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

from translator.epub_doc import list_chapters

from .engine import VitsEngine
from .pipeline import concat_chapter_mp3s, process_chapter

DEFAULT_CKPT = "models/vits/G_100000.pth"
DEFAULT_VITS = "third_party/vits"


def cmd_list(args: argparse.Namespace) -> None:
    import re
    import zipfile

    book_path = Path(args.epub)
    book = list_chapters(book_path)
    dev_re = re.compile(r"[\u0900-\u097F]")
    lat_re = re.compile(r"[A-Za-z]")
    with zipfile.ZipFile(book_path) as z:
        print(f"\n{args.epub}  -  {len(book.chapters)} spine documents\n")
        print("  #  chars   lang   title")
        print("  " + "-" * 66)
        for ch in book.chapters:
            try:
                data = z.read(ch.href).decode("utf-8", errors="replace")
            except KeyError:
                print(f"  {ch.file_index:2d}  {ch.chars:6d}   ???   {ch.title or ch.href}")
                continue
            dev, lat = len(dev_re.findall(data)), len(lat_re.findall(data))
            lang = "ne" if dev > 200 and dev >= lat // 2 else "en"
            print(f"  {ch.file_index:2d}  {ch.chars:6d}   {lang}     "
                  f"{ch.title or ch.href}")
    print("\n  lang: ne = Devanagari (ready for TTS), en = still English")


def cmd_pause(args: argparse.Namespace) -> None:
    flag = Path(args.dir) / "PAUSE"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(str(time.time()), encoding="utf-8")
    print(f"Pause requested: {flag}")
    print("The running `tts run` will stop after its current chunk, write the "
          "partial chapter audio and unload the model.")


def _select_chapters(book, args) -> list:
    if args.all:
        return list(book.chapters)
    if not args.chapters:
        sys.exit("error: use --chapters 1,2,3 or --all (see `list`)")
    wanted = []
    for part in str(args.chapters).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            wanted.extend(range(int(a), int(b) + 1))
        else:
            wanted.append(int(part))
    sel = [c for c in book.chapters if c.file_index in wanted]
    missing = set(wanted) - {c.file_index for c in sel}
    if missing:
        sys.exit(f"error: no such chapter index: {sorted(missing)}")
    return sel


def cmd_run(args: argparse.Namespace) -> None:
    book_path = Path(args.epub)
    if not book_path.exists():
        sys.exit(f"error: {book_path} not found")
    if not Path(args.checkpoint).exists():
        sys.exit(f"error: checkpoint {args.checkpoint} not found - "
                 f"run `python scripts\\download_vits.py` first")
    book = list_chapters(book_path)
    chapters = _select_chapters(book, args)

    out_dir, pause_flag, engine, deadline = _prepare_run(args)

    print(f"\ncheckpoint : {args.checkpoint}")
    print(f"chapters   : {', '.join(str(c.file_index) for c in chapters)}"
          f"  ({len(chapters)} of {len(book.chapters)})")
    print(f"output dir : {out_dir}")
    print(f"pause      : `python -m tts pause --dir {out_dir}`"
          f"{', or after ' + str(args.pause_after_min) + ' min' if args.pause_after_min else ''}"
          f", or Ctrl+C\n")

    t_start = time.time()
    results = []
    paused = False

    try:
        for ch in chapters:
            res = process_chapter(
                book, ch, engine, out_dir,
                max_chars=args.max_chars,
                gap_sentence=args.gap_sentence, gap_para=args.gap_para,
                gap_heading=args.gap_heading,
                pause_flag=pause_flag, deadline=deadline,
                no_resume=args.no_resume, no_mp3=args.no_mp3)
            results.append(res)
            if res.chunks:
                print(f"  -> {res.chunks} chunks ({res.resumed} cached, "
                      f"{res.synthesized} new) | {res.audio_seconds:.0f}s audio "
                      f"in {res.seconds:.1f}s "
                      f"(x{res.realtime_factor:.1f} realtime)"
                      f"{' | PAUSED' if res.paused else ''}")
                if res.output:
                    print(f"     audio: {res.output}")
                print()
            else:
                print("  -> no Nepali text found (chapter not translated yet?)\n")
            if res.paused:
                paused = True
                break
    except KeyboardInterrupt:
        paused = True
        print("\nInterrupted - progress checkpointed per chunk; "
              "re-run the same command to resume and assemble.\n")

    total = time.time() - t_start
    total_chunks = sum(r.chunks for r in results)
    total_audio = sum(r.audio_seconds for r in results)
    total_synth = sum(r.synthesized for r in results)
    total_resume = sum(r.resumed for r in results)
    total_seconds = sum(r.seconds for r in results)

    print("=" * 74)
    status = "PAUSED (partial)" if paused else "COMPLETE"
    print(f"status              : {status}")
    outputs = [r.output for r in results if r.output]
    for o in outputs:
        print(f"audio               : {o}")
    print(f"chunks              : {total_chunks} total "
          f"({total_resume} restored, {total_synth} synthesized)")
    if total_audio:
        print(f"audio length        : {total_audio / 60:.1f} min")
    if total_seconds > 0.5:
        print(f"synthesis time      : {total_seconds:.1f}s "
              f"({total_seconds / 60:.1f} min)   total wall: {total:.1f}s")
        if total_synth:
            print(f"speed               : {total_audio / total_seconds:.1f}x "
                  f"realtime ({total_synth / total_seconds:.2f} chunks/s)")
    warns = sum(len(r.warnings) for r in results)
    if warns:
        print(f"warnings            : {warns} (see tts_report.json)")
    if not paused and not args.no_concat and len(outputs) > 1:
        full = concat_chapter_mp3s(out_dir, Path(book_path).stem)
        if full:
            print(f"full book           : {full}")
    if paused:
        print(f"\nresume with the same command; progress is checkpointed in "
              f"{out_dir / 'audio'}")
    print("=" * 74)

    import json
    (out_dir / "tts_report.json").write_text(
        json.dumps({
            "epub": str(book_path), "status": status,
            "checkpoint": args.checkpoint,
            "chapters": [asdict(r) for r in results],
            "totals": {"chunks": total_chunks, "synthesized": total_synth,
                       "restored": total_resume,
                       "audio_seconds": round(total_audio, 2),
                       "synth_seconds": round(total_seconds, 2),
                       "wall_seconds": round(total, 2)},
            "outputs": outputs,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"report saved to {out_dir / 'tts_report.json'}")

    if args.keep_model:
        print("model left loaded (--keep-model)")
    else:
        engine.free()
        print("model unloaded from VRAM")


def _add_synth_options(p: argparse.ArgumentParser) -> None:
    """Options shared by `run` (EPUB) and `text` (plain .txt)."""
    p.add_argument("--out", default="output",
                   help="output directory (default: output)")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT,
                   help=f"VITS checkpoint (default: {DEFAULT_CKPT})")
    p.add_argument("--vits-dir", default=DEFAULT_VITS,
                   help=f"cloned VITS code (default: {DEFAULT_VITS})")
    p.add_argument("--device", default="cuda",
                   help="cuda or cpu (default: cuda, falls back to cpu)")
    p.add_argument("--max-chars", type=int, default=180,
                   help="max chars per synthesized chunk (default 180); "
                        "long inputs degrade speech quality")
    p.add_argument("--length-scale", type=float, default=1.0,
                   help="speech speed: <1 faster, >1 slower (default 1.0)")
    p.add_argument("--noise-scale", type=float, default=0.667)
    p.add_argument("--noise-scale-w", type=float, default=0.8)
    p.add_argument("--gap-sentence", type=float, default=0.25,
                   help="silence between sentences in seconds (default 0.25)")
    p.add_argument("--gap-para", type=float, default=0.65,
                   help="silence between paragraphs (default 0.65)")
    p.add_argument("--gap-heading", type=float, default=1.0,
                   help="silence before a heading (default 1.0)")
    p.add_argument("--pause-after-min", type=float, default=0,
                   help="pause automatically after N minutes")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore checkpoints and re-synthesize everything")
    p.add_argument("--no-mp3", action="store_true",
                   help="keep WAV instead of encoding MP3")
    p.add_argument("--no-concat", action="store_true",
                   help="skip the full-book MP3 merge")
    p.add_argument("--keep-model", action="store_true",
                   help="do not unload the model when finished")


def _prepare_run(args) -> tuple[Path, Path, "VitsEngine", float | None]:
    """Common setup for run/text: out dir, stale-flag cleanup, engine, deadline."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pause_flag = out_dir / "PAUSE"
    if pause_flag.exists():  # stale flag from an old session: don't pause instantly
        pause_flag.unlink()
        print("note: removed a stale PAUSE flag")
    engine = VitsEngine(Path(args.vits_dir), Path(args.checkpoint),
                        device=args.device, noise_scale=args.noise_scale,
                        noise_scale_w=args.noise_scale_w,
                        length_scale=args.length_scale)
    deadline = (time.time() + args.pause_after_min * 60
                if args.pause_after_min else None)
    return out_dir, pause_flag, engine, deadline


def _write_report(out_dir: Path, payload: dict) -> None:
    import json
    (out_dir / "tts_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"report saved to {out_dir / 'tts_report.json'}")


def cmd_text(args: argparse.Namespace) -> None:
    """Synthesize speech from a plain Devanagari .txt file."""
    from .pipeline import load_text_paragraphs, synthesize_paragraphs

    txt_path = Path(args.txt)
    if not txt_path.exists():
        sys.exit(f"error: {txt_path} not found")
    if not Path(args.checkpoint).exists():
        sys.exit(f"error: checkpoint {args.checkpoint} not found - "
                 f"run `python scripts\\download_vits.py` first")
    paragraphs = load_text_paragraphs(txt_path)
    if not paragraphs:
        sys.exit(f"error: {txt_path} contains no text")

    out_dir, pause_flag, engine, deadline = _prepare_run(args)
    tag = txt_path.stem          # -> output/audio/<stem>.mp3 + <stem>/manifest

    print(f"\ninput      : {txt_path}  ({len(paragraphs)} paragraphs, "
          f"{sum(len(p.text) for p in paragraphs):,} chars)")
    print(f"checkpoint : {args.checkpoint}")
    print(f"output     : {out_dir / 'audio' / (tag + '.mp3')}")
    print(f"pause      : `python -m tts pause --dir {out_dir}`"
          f"{', or after ' + str(args.pause_after_min) + ' min' if args.pause_after_min else ''}"
          f", or Ctrl+C\n")

    t_start = time.time()
    res = None
    try:
        res = synthesize_paragraphs(
            tag, txt_path.name, paragraphs, engine, out_dir,
            max_chars=args.max_chars, gap_sentence=args.gap_sentence,
            gap_para=args.gap_para, gap_heading=args.gap_heading,
            pause_flag=pause_flag, deadline=deadline,
            no_resume=args.no_resume, no_mp3=args.no_mp3)
    except KeyboardInterrupt:
        print("\nInterrupted - progress checkpointed per chunk; "
              "re-run the same command to resume and assemble.\n")

    total = time.time() - t_start
    print("=" * 74)
    status = "PAUSED (partial)" if (res is None or res.paused) else "COMPLETE"
    print(f"status              : {status}")
    if res:
        print(f"audio               : {res.output}")
        print(f"chunks              : {res.chunks} "
              f"({res.resumed} restored, {res.synthesized} synthesized)")
        if res.audio_seconds:
            print(f"audio length        : {res.audio_seconds / 60:.1f} min")
        if res.seconds > 0.5:
            print(f"synthesis time      : {res.seconds:.1f}s   "
                  f"total wall: {total:.1f}s")
            if res.synthesized:
                print(f"speed               : {res.audio_seconds / res.seconds:.1f}x "
                      f"realtime")
        if res.warnings:
            print(f"warnings            : {len(res.warnings)} (see tts_report.json)")
    if status == "PAUSED (partial)":
        print(f"\nresume with the same command; progress is checkpointed in "
              f"{out_dir / 'audio' / tag}")
    print("=" * 74)

    from dataclasses import asdict
    _write_report(out_dir, {
        "input": str(txt_path), "status": status,
        "checkpoint": args.checkpoint,
        "chapters": [asdict(res)] if res else [],
        "totals": {"chunks": res.chunks if res else 0,
                   "synthesized": res.synthesized if res else 0,
                   "restored": res.resumed if res else 0,
                   "audio_seconds": round(res.audio_seconds, 2) if res else 0,
                   "wall_seconds": round(total, 2)},
        "outputs": [res.output] if res and res.output else [],
    })

    if args.keep_model:
        print("model left loaded (--keep-model)")
    else:
        engine.free()
        print("model unloaded from VRAM")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tts", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list chapters (ne = ready for TTS)")
    p_list.add_argument("epub")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="synthesize speech (resumable)")
    p_run.add_argument("epub", help="translated (Devanagari) EPUB")
    p_run.add_argument("--chapters", help="e.g. 1,3,5-8 (1-based spine order)")
    p_run.add_argument("--all", action="store_true", help="every chapter")
    _add_synth_options(p_run)
    p_run.set_defaults(func=cmd_run)

    p_text = sub.add_parser(
        "text", help="synthesize speech from a plain Devanagari .txt file")
    p_text.add_argument("txt", help="UTF-8 text file you translated yourself")
    _add_synth_options(p_text)
    p_text.set_defaults(func=cmd_text)

    p_pause = sub.add_parser("pause", help="ask a running synthesis to pause")
    p_pause.add_argument("--dir", default="output",
                         help="the run's --out directory (default: output)")
    p_pause.set_defaults(func=cmd_pause)

    args = ap.parse_args(argv)
    if args.cmd == "run" and not (args.chapters or args.all):
        ap.error("run requires --chapters N,N or --all")
    args.func(args)


if __name__ == "__main__":
    main()
