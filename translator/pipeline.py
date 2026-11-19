"""End-to-end translation pipeline: extract -> group -> translate (checkpointed)
-> rebuild EPUB.

Design notes:
- Paragraph groups (not isolated sentences) are the unit of translation, sized
  adaptively from the chapter's total length.
- Every LLM request is stateless: fresh context, only the last translated
  paragraph is carried over (truncated), so context never accumulates and the
  model cannot drift/hallucinate from stale history.
- Progress is checkpointed after EVERY group, so the run can be paused at any
  time (PAUSE flag file, --pause-after-min, or Ctrl+C) and resumed later; on
  pause a partial EPUB with everything translated so far is written.
- The model is explicitly unloaded from VRAM when the run ends (or pauses).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .epub_doc import (EXTRACT_VERSION, Chapter, Group, Unit, extract_units,
                       list_chapters, merge_units)
from .llm import OllamaClient, translate_batch

SYSTEM_TEMPLATE = """You are a professional literary translator translating the book "{book}" (English -> Nepali), chapter: "{chapter}".

Rules:
- Translate into natural, fluent literary Nepali in Devanagari script.
- Preserve the author's tone, humor and register (philosophy novel for young readers).
- Each request is INDEPENDENT and complete: translate ONLY the text given, never continue, never add, never summarize.
- Return a JSON array with exactly the same number of items as requested, same order.
- Segments are consecutive parts of the same paragraph: keep pronouns and terminology consistent across them.
- Transliterate proper names consistently (Sophie -> सोफी, Hilde -> हिल्डे, Alberto -> अल्बर्डो, Hermes -> हर्मिस).
- Reply with ONLY the JSON array - no explanations, no markdown.
{glossary_block}"""

GLOSSARY_DEFAULT = """- Sophie -> सोफी
- Hilde -> हिल्डे
- Alberto Knag -> अल्बर्डो क्नाग
- Hermes -> हर्मिस
- Sophie's World -> सोफीको संसार
- philosophy -> दर्शनशास्त्र
- philosopher -> दार्शनिक
"""


@dataclass
class ChapterResult:
    chapter: Chapter
    units: int
    translated: int
    seconds: float
    resumed: int = 0
    requests: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    groups: int = 0
    paused: bool = False


# translations kept in memory for the rebuild stage (same process run)
result_translations: dict[int, dict[int, str]] = {}


def _checkpoint_path(cp_dir: Path, epub_name: str, chapter: Chapter) -> Path:
    stem = Path(epub_name).stem
    safe = Path(chapter.href).name.replace(".", "_")
    return cp_dir / f"{stem}__{safe}.json"


def load_checkpoint(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("meta", {}).get("v") != EXTRACT_VERSION:
            return {}  # extraction rules changed -> old indexes are invalid
        return {int(k): v for k, v in data.get("translations", {}).items()}
    except (json.JSONDecodeError, OSError, ValueError):
        return {}


def save_checkpoint(path: Path, meta: dict, translations: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"v": EXTRACT_VERSION, **meta}
    path.write_text(
        json.dumps({"meta": meta,
                    "translations": {str(k): v for k, v in sorted(translations.items())}},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")


def _glossary_block(path: Path | None) -> str:
    if path and path.exists():
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            return "Glossary (always use these renderings):\n" + "\n".join(f"- {l}" for l in lines) + "\n"
    return "Key terms:\n" + GLOSSARY_DEFAULT


def adaptive_grouping(units: list[Unit], par_chars: int | None) -> tuple[int, int]:
    """Pick paragraph-group target size from the chapter's total text length."""
    total = sum(len(u.source) for u in units)
    if par_chars:
        target = par_chars
    else:
        # ~20 paragraph-sized groups per chapter, clamped to a sane band
        target = max(400, min(1200, total // 20)) if units else 800
    maximum = max(800, min(2400, target * 2))
    return target, maximum


def translate_chapter(client: OllamaClient, book_path: Path, chapter: Chapter,
                      out_dir: Path, glossary: str, par_chars: int | None = None,
                      resume: bool = True, prev_context_chars: int = 400,
                      pause_flag: Path | None = None,
                      deadline: float | None = None, log=print) -> ChapterResult:
    units = extract_units(book_path, chapter)
    target, maximum = adaptive_grouping(units, par_chars)
    groups = merge_units(units, target=target, maximum=maximum)

    cp = _checkpoint_path(out_dir / "checkpoints", book_path.name, chapter)
    translations = load_checkpoint(cp) if resume else {}
    resumed_units = sum(1 for u in units if u.unit_index in translations)
    todo = [g for g in groups if not all(u.unit_index in translations for u in g.units)]

    log(f"  [{chapter.file_index:02d}] {chapter.title or chapter.href}: "
        f"{len(units)} segments -> {len(groups)} paragraph groups "
        f"(target {target} chars), {resumed_units} cached, {len(todo)} groups to go")

    system = SYSTEM_TEMPLATE.format(book=book_path.stem.replace("_", " "),
                                    chapter=chapter.title or chapter.href,
                                    glossary_block=glossary)

    t0 = time.time()
    stats_start = dict(client.stats.__dict__)
    prev_translation = ""
    done_units = 0
    paused = False

    for gi, group in enumerate(todo):
        # --- pause checks (flag file or time budget) ---
        if pause_flag is not None and pause_flag.exists():
            paused = True
            break
        if deadline is not None and time.time() > deadline:
            paused = True
            break

        texts = [u.source for u in group.units]
        if group.heading:
            # A heading split into spans is ONE title: translate it as a single
            # string so it never renders as mixed English/Devanagari fragments.
            whole = " ".join(texts)
            out = translate_batch(client, system, [whole], prev=prev_translation,
                                  full_text=whole)
            if out and out[0].strip():
                translations[group.units[0].unit_index] = out[0].strip()
                for u in group.units[1:]:
                    translations[u.unit_index] = ""
                prev_translation = out[0].strip()[:prev_context_chars]
        else:
            out = translate_batch(client, system, texts, prev=prev_translation,
                                  full_text=group.text)
            if out is None:
                # fallback: per-segment with the full paragraph as context
                out = []
                for t in texts:
                    single = translate_batch(client, system, [t], prev=prev_translation)
                    out.append(single[0] if single else t)
                log(f"    group {gi + 1}/{len(todo)} fell back to per-segment mode")
            for u, tr in zip(group.units, out):
                if tr and tr.strip():
                    translations[u.unit_index] = tr.strip()
            if out:
                prev_translation = out[-1][:prev_context_chars]
        done_units += len(group.units)
        save_checkpoint(cp, {"epub": book_path.name, "chapter": chapter.title},
                        translations)
        # expose partial progress for the rebuild step immediately
        result_translations[chapter.file_index] = translations

        if (gi + 1) % 5 == 0 or gi + 1 == len(todo):
            el = time.time() - t0
            rate = (gi + 1) / el if el else 0
            eta = (len(todo) - gi - 1) / rate if rate else 0
            log(f"    {gi + 1}/{len(todo)} groups ({done_units} segments) "
                f"{el:.0f}s elapsed, ETA {eta:.0f}s")

    if pause_flag is not None and pause_flag.exists():
        try:
            pause_flag.unlink()
        except OSError:
            pass

    elapsed = time.time() - t0
    bs = client.stats
    result_translations[chapter.file_index] = translations
    return ChapterResult(
        chapter=chapter, units=len(units),
        translated=sum(1 for u in units if u.unit_index in translations),
        seconds=elapsed, resumed=resumed_units, groups=len(groups),
        requests=bs.requests - stats_start["requests"],
        retries=bs.retries - stats_start["retries"],
        prompt_tokens=bs.prompt_tokens - stats_start["prompt_tokens"],
        output_tokens=bs.output_tokens - stats_start["output_tokens"],
        paused=paused,
    )


def build_epub(book_path: Path, chapters: list[Chapter], out_path: Path) -> int:
    """Rebuild the full EPUB, applying whatever translations are in memory.

    Chapters without translations are copied unchanged, so the output is always
    a complete, openable book (translated so far). Returns applied-chapter count.
    """
    import zipfile
    from . import epub_doc

    with zipfile.ZipFile(book_path) as z:
        infos = z.infolist()
        blobs = {i.filename: z.read(i.filename) for i in infos}

    applied = 0
    for ch in chapters:
        tr = result_translations.get(ch.file_index)
        if not tr:
            continue
        applied += 1
        tree = epub_doc.parse_xml(blobs[ch.href])
        doctype = tree.docinfo.doctype or ""
        counter = {"i": 0}

        def handle(el, kind, _tr=tr, _counter=counter):
            text = el.text if kind == "text" else el.tail
            if text is None or not epub_doc._is_translatable(text):
                return
            parent_tag = (el.tag.split("}")[-1] if isinstance(el.tag, str) else "?")
            if parent_tag in epub_doc.SKIP_TAGS or epub_doc._skip_zone(el):
                return
            idx = _counter["i"]
            _counter["i"] += 1
            if idx not in _tr:
                return
            new = _tr[idx]  # may be "" (collapsed heading segment)
            lead = text[: len(text) - len(text.lstrip())]
            trail = text[len(text.rstrip()):]
            if kind == "text":
                el.text = lead + new + trail
            else:
                el.tail = lead + new + trail

        for el in tree.iter():
            if not isinstance(el.tag, str):
                continue
            if el.text:
                handle(el, "text")
            if el.tail:
                handle(el, "tail")

        blobs[ch.href] = epub_doc.etree.tostring(
            tree, encoding="UTF-8", xml_declaration=True, doctype=doctype or None)

    epub_doc._write_zip(book_path, out_path, infos, blobs)
    return applied


def save_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
