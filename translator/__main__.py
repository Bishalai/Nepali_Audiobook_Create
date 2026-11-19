"""CLI: python -m translator <command>

Commands:
  list   <epub>                       list chapters
  run    <epub> --chapters 1,3        translate selected chapters (1-based spine order)
  run    <epub> --all                 translate the whole book
  pause                                ask a running `run` to pause gracefully
  unload                               eject the model from VRAM

Pause / resume:
  * `python -m translator pause`      writes a PAUSE flag the running job checks
                                      after every paragraph group
  * Ctrl+C                            also pauses safely
  * `--pause-after-min N`             automatic pause after N minutes
  Every pause writes `<book>_ne_partial.epub` with everything translated so far;
  re-run the same `run` command to resume from the checkpoints.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .epub_doc import list_chapters
from .llm import OllamaClient
from .pipeline import (build_epub, result_translations, save_report,
                       translate_chapter, _glossary_block)

DEFAULT_MODEL = "qwen3.5:9b-q4_K_M"


def cmd_list(args: argparse.Namespace) -> None:
    book = list_chapters(Path(args.epub))
    print(f"\n{args.epub}  -  {len(book.chapters)} spine documents\n")
    print("  #  chars   title")
    print("  " + "-" * 66)
    for ch in book.chapters:
        print(f"  {ch.file_index:2d}  {ch.chars:6d}   {ch.title or ch.href}")


def cmd_pause(args: argparse.Namespace) -> None:
    flag = Path(args.dir) / "PAUSE"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text(str(time.time()), encoding="utf-8")
    print(f"Pause requested: {flag}")
    print("The running `translator run` will stop after its current paragraph "
          "group, write a partial EPUB and unload the model.")


def cmd_unload(args: argparse.Namespace) -> None:
    ok = OllamaClient(args.model).unload()
    print("Model unloaded from VRAM." if ok else "Could not reach Ollama (already stopped?)")


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
    book = list_chapters(book_path)
    chapters = _select_chapters(book, args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pause_flag = out_dir / "PAUSE"
    if pause_flag.exists():  # stale flag from an old session: don't pause instantly
        pause_flag.unlink()
        print("note: removed a stale PAUSE flag")

    glossary = _glossary_block(Path(args.glossary) if args.glossary else None)
    client = OllamaClient(model=args.model, temperature=args.temperature,
                          num_ctx=args.num_ctx, num_predict=args.num_predict)
    deadline = (time.time() + args.pause_after_min * 60
                if args.pause_after_min else None)

    print(f"\nmodel      : {args.model}")
    print(f"chapters   : {', '.join(str(c.file_index) for c in chapters)}"
          f"  ({len(chapters)} of {len(book.chapters)})")
    print(f"output dir : {out_dir}")
    print(f"pause      : `python -m translator pause --dir {out_dir}`"
          f"{', or after ' + str(args.pause_after_min) + ' min' if args.pause_after_min else ''}"
          f", or Ctrl+C\n")

    t_start = time.time()
    results = []
    paused = interrupted = False

    try:
        for ch in chapters:
            res = translate_chapter(
                client, book_path, ch, out_dir, glossary,
                par_chars=args.par_chars if args.par_chars else None,
                resume=not args.no_resume,
                prev_context_chars=args.prev_context_chars,
                pause_flag=pause_flag, deadline=deadline)
            results.append(res)
            if res.requests:
                print(f"  -> {res.translated}/{res.units} segments in {res.seconds:.1f}s | "
                      f"{res.requests} requests | {res.prompt_tokens}+{res.output_tokens} tokens"
                      f"{' | PAUSED' if res.paused else ''}\n")
            else:
                print(f"  -> fully restored from checkpoint (0 requests)\n")
            if res.paused:
                paused = True
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted - checkpoint saved, writing partial EPUB...")

    # ---------- rebuild EPUB (full or translated-so-far) ----------
    partial = paused or interrupted
    suffix = "_all" if args.all else "_ch" + "_".join(
        str(r.chapter.file_index) for r in results)
    out_epub = out_dir / f"{book_path.stem}_ne{suffix}{'_partial' if partial else ''}.epub"
    applied = 0 if args.no_epub else build_epub(book_path, book.chapters, out_epub)

    total = time.time() - t_start
    total_units = sum(r.units for r in results)
    total_done = sum(r.translated for r in results)
    tr_time = sum(r.seconds for r in results)
    src_chars = _source_chars(book, results)

    print("=" * 74)
    status = "PAUSED (partial)" if partial else "COMPLETE"
    print(f"status              : {status}")
    print(f"translated EPUB     : {out_epub}  ({applied} chapter(s) translated, "
          f"rest unchanged English)")
    print(f"segments            : {total_done}/{total_units}")
    print(f"source chars        : {src_chars:,}")
    if tr_time > 0.5:
        print(f"translate time      : {tr_time:.1f}s ({tr_time / 60:.1f} min)   "
              f"total wall: {total:.1f}s")
        print(f"throughput          : {src_chars / tr_time:.0f} chars/s | "
              f"{total_units / tr_time:.2f} segments/s")
        if not partial:
            rest = sum(c.chars for c in book.chapters
                       if c.file_index not in {r.chapter.file_index for r in results})
            rate = src_chars / tr_time
            print(f"whole-book ETA      : {rest:,} chars remaining -> "
                  f"~{rest / rate / 60:.0f} min ({rest / rate / 3600:.1f} h)")
    else:
        print(f"translate time      : 0 LLM requests (everything from checkpoints), "
              f"wall {total:.1f}s")
    print(f"tokens              : {client.stats.prompt_tokens} prompt / "
          f"{client.stats.output_tokens} output, {client.stats.retries} retries, "
          f"{client.stats.failures} failures")
    if partial:
        print(f"\nresume with the same command; progress is checkpointed in "
              f"{out_dir / 'checkpoints'}")
    print("=" * 74)

    save_report(out_dir / "report.json", {
        "epub": str(book_path), "model": args.model, "status": status,
        "chapters": [{
            "index": r.chapter.file_index, "title": r.chapter.title,
            "units": r.units, "translated": r.translated, "resumed": r.resumed,
            "groups": r.groups, "seconds": round(r.seconds, 2),
            "requests": r.requests, "retries": r.retries,
            "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens,
            "paused": r.paused,
        } for r in results],
        "totals": {"translate_seconds": round(tr_time, 2),
                   "wall_seconds": round(total, 2),
                   "source_chars": src_chars, "units": total_units,
                   "translated": total_done,
                   "prompt_tokens": client.stats.prompt_tokens,
                   "output_tokens": client.stats.output_tokens},
        "output_epub": str(out_epub),
    })
    print(f"report saved to {out_dir / 'report.json'}")

    # ---------- eject the model automatically ----------
    if args.keep_model:
        print("model left loaded (--keep-model)")
    else:
        if client.unload():
            print("model unloaded from VRAM (VRAM freed)")
        else:
            print("note: could not unload model (Ollama not reachable)")


def _source_chars(book, results) -> int:
    """Source characters of the chapters we (attempted to) translate."""
    from .epub_doc import extract_units
    total = 0
    seen = set()
    for r in results:
        if r.chapter.file_index in seen:
            continue
        seen.add(r.chapter.file_index)
        total += sum(len(u.source) for u in extract_units(book.path, r.chapter))
    return total


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="translator", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list chapters in the EPUB spine")
    p_list.add_argument("epub")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="translate chapters (resumable)")
    p_run.add_argument("epub")
    p_run.add_argument("--chapters", help="e.g. 1,3,5-8 (1-based spine order)")
    p_run.add_argument("--all", action="store_true", help="translate every chapter")
    p_run.add_argument("--out", default="output", help="output directory (default: output)")
    p_run.add_argument("--model", default=DEFAULT_MODEL)
    p_run.add_argument("--glossary", help="glossary file, one 'term -> नेपाली' per line")
    p_run.add_argument("--temperature", type=float, default=0.3)
    p_run.add_argument("--num-ctx", type=int, default=4096,
                       help="fresh context window per request (default 4096)")
    p_run.add_argument("--num-predict", type=int, default=2048)
    p_run.add_argument("--par-chars", type=int, default=0,
                       help="paragraph-group target size in chars "
                            "(default: adaptive = chapter_chars/20, 400-1200)")
    p_run.add_argument("--prev-context-chars", type=int, default=400,
                       help="max chars of previous paragraph carried as context (default 400)")
    p_run.add_argument("--pause-after-min", type=float, default=0,
                       help="pause automatically after N minutes")
    p_run.add_argument("--keep-model", action="store_true",
                       help="do not unload the model when finished")
    p_run.add_argument("--no-resume", action="store_true",
                       help="ignore checkpoints and re-translate")
    p_run.add_argument("--no-epub", action="store_true",
                       help="translate only, skip EPUB rebuild")
    p_run.set_defaults(func=cmd_run)

    p_pause = sub.add_parser("pause", help="ask a running translation to pause")
    p_pause.add_argument("--dir", default="output",
                         help="the run's --out directory (default: output)")
    p_pause.set_defaults(func=cmd_pause)

    p_unload = sub.add_parser("unload", help="eject the model from VRAM")
    p_unload.add_argument("--model", default=DEFAULT_MODEL)
    p_unload.set_defaults(func=cmd_unload)

    args = ap.parse_args(argv)
    if args.cmd == "run" and not (args.chapters or args.all):
        ap.error("run requires --chapters N,N or --all")
    args.func(args)


if __name__ == "__main__":
    main()
