"""TTS pipeline: translated EPUB -> sentence chunks -> per-chunk WAVs (checkpointed)
-> assembled chapter MP3. Pause/resume mirrors the translation CLI.

- Numbers are spelled out in Nepali (tts.numbers) before synthesis.
- Chunks are sentence-sized (default <= 180 chars) because VITS quality
  degrades on long inputs.
- Every chunk's WAV is written immediately; manifest.json records status so
  runs resume exactly where they stopped.
- PAUSE flag file or Ctrl+C -> partial chapter audio + graceful exit.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from translator.epub_doc import SKIP_TAGS, Chapter, EpubBook, parse_xml

from .engine import VitsEngine
from .prep import Chunk, Paragraph, chunk_paragraph, has_devanagari

PARA_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote",
             "td", "th", "dd", "dt", "figcaption", "caption", "pre", "div",
             "section", "article"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
MANIFEST_VERSION = 3   # v3: chunk text keeps danda (sentence splits for video)
_LATIN_RE = None


def _has_latin(text: str) -> bool:
    global _LATIN_RE
    if _LATIN_RE is None:
        import re
        _LATIN_RE = re.compile(r"[A-Za-z]{2,}")
    return bool(_LATIN_RE.search(text))


def extract_paragraphs(book_path: Path, chapter: Chapter) -> tuple[list[Paragraph], dict]:
    """Leaf block elements in document order, skipping untranslated ones."""
    import zipfile

    with zipfile.ZipFile(book_path) as z:
        tree = parse_xml(z.read(chapter.href))

    paragraphs: list[Paragraph] = []
    stats = {"found": 0, "skipped_english": 0, "empty": 0}

    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.split("}")[-1]
        if tag not in PARA_TAGS:
            continue
        # skip zones (script/style/head/...)
        skip = False
        for anc in el.iterancestors():
            if isinstance(anc.tag, str) and anc.tag.split("}")[-1] in SKIP_TAGS:
                skip = True
                break
        if skip:
            continue
        # only leaf blocks (a <div> containing <p>s is not its own paragraph).
        # NOTE: el.iter() yields el itself - must use iterdescendants() here,
        # otherwise every element looks like it has a block child.
        has_block_child = any(
            isinstance(c.tag, str) and c.tag.split("}")[-1] in PARA_TAGS
            for c in el.iterdescendants())
        if has_block_child:
            continue
        raw = "".join(el.itertext())
        if not raw.strip():
            continue
        stats["found"] += 1
        if not has_devanagari(raw) and _has_latin(raw):
            stats["skipped_english"] += 1  # chapter (or paragraph) not translated yet
            continue
        kind = "heading" if tag in HEADING_TAGS else "para"
        paragraphs.append(Paragraph(kind=kind, text=raw))
    stats["empty"] = stats["found"] - stats["skipped_english"] - len(paragraphs)
    return paragraphs, stats


def build_chunks(paragraphs: list[Paragraph], vits_dir: Path, max_chars: int,
                 gap_sentence: float, gap_para: float, gap_heading: float,
                 warnings: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pi, para in enumerate(paragraphs):
        chunks.extend(chunk_paragraph(
            pi, para.kind, para.text, vits_dir, max_chars=max_chars,
            gap_sentence=gap_sentence, gap_para=gap_para,
            gap_heading=gap_heading, warnings=warnings))
    for i, c in enumerate(chunks):
        c.i = i
    return chunks


def source_hash(paragraphs: list[Paragraph]) -> str:
    """Hash of the input text - lets `make_video` detect an edited .txt."""
    import hashlib
    joined = "\n\u241e".join(p.text for p in paragraphs)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:20]


def load_text_paragraphs(txt_path: Path) -> list[Paragraph]:
    """Plain UTF-8 Devanagari .txt -> paragraph list.

    - blank lines separate paragraphs (preferred);
    - a file with no blank lines but many short, sentence-terminated lines
      is treated as one-sentence-per-line (each line = one paragraph);
    - anything else is joined into a single flowing block (the chunker
      re-splits it at sentence boundaries).
    Short first/titelike lines become headings (longer silence before them).
    """
    import re

    raw = txt_path.read_text(encoding="utf-8-sig")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []

    def is_heading(t: str) -> bool:
        return len(t) <= 60 and not any(c in t for c in "।?!.")

    if re.search(r"\n[ \t]*\n", raw):
        blocks = [b.strip() for b in re.split(r"\n[ \t]*\n+", raw) if b.strip()]
    else:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        terminals = "।?!.'\""
        terminated = sum(1 for l in lines if l and l[-1] in terminals)
        line_per_sentence = (len(lines) >= 3
                             and sum(len(l) for l in lines) / len(lines) < 130
                             and terminated / len(lines) >= 0.6)
        blocks = lines if line_per_sentence else [" ".join(lines)]

    paragraphs: list[Paragraph] = []
    for i, b in enumerate(blocks):
        # inside a block, single newlines are soft wraps -> spaces
        text = " ".join(b.split())
        if not text:
            continue
        kind = "heading" if (i > 0 and is_heading(text)) else "para"
        paragraphs.append(Paragraph(kind=kind, text=text))
    return paragraphs


def _manifest_path(chapter_dir: Path) -> Path:
    return chapter_dir / "manifest.json"


def load_manifest(chapter_dir: Path) -> dict:
    p = _manifest_path(chapter_dir)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("v") == MANIFEST_VERSION:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"v": MANIFEST_VERSION, "chunks": {}}


def save_manifest(chapter_dir: Path, manifest: dict) -> None:
    _manifest_path(chapter_dir).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


def resume_chunks(chapter_dir: Path, chunks: list[Chunk]) -> int:
    """Mark chunks done when the recorded hash matches and the WAV exists."""
    manifest = load_manifest(chapter_dir)
    saved: dict = manifest.get("chunks", {})
    n = 0
    for c in chunks:
        rec = saved.get(str(c.i))
        wav = chapter_dir / c.wav_name
        if rec and rec.get("hash") == c.hash and wav.exists():
            c.status = "done"
            c.file = c.wav_name
            c.audio_seconds = float(rec.get("audio_seconds", 0.0))
            n += 1
    return n


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    wavfile.write(path, sr, (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16))


def assemble(chapter_dir: Path, chunks: list[Chunk], sr: int,
             out_wav: Path) -> tuple[float, int]:
    """Concatenate done chunk WAVs with their silence gaps. Returns (seconds, count)."""
    pieces: list[np.ndarray] = []
    used = 0
    for c in chunks:
        if c.status != "done":
            continue
        wav = chapter_dir / c.wav_name
        if not wav.exists():
            continue
        if c.gap_before > 0 and pieces:
            pieces.append(np.zeros(int(c.gap_before * sr), dtype=np.float32))
        _, data = wavfile.read(wav)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32767.0
        else:
            data = data.astype(np.float32)
        pieces.append(data)
        used += 1
    if not pieces:
        return 0.0, 0
    audio = np.concatenate(pieces)
    write_wav(out_wav, audio, sr)
    return len(audio) / sr, used


def to_mp3(wav_path: Path, mp3_path: Path, keep_wav: bool = False) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(wav_path),
         "-codec:a", "libmp3lame", "-b:a", "96k", "-ac", "1",
         str(mp3_path)],
        check=True)
    if not keep_wav:
        wav_path.unlink(missing_ok=True)
    return True


@dataclass
class ChapterTTSResult:
    index: int
    title: str
    chunks: int
    synthesized: int
    resumed: int
    skipped_english: int
    seconds: float
    audio_seconds: float
    realtime_factor: float
    output: str
    paused: bool = False
    warnings: list[str] = field(default_factory=list)


def synthesize_paragraphs(tag: str, title: str, paragraphs: list[Paragraph],
                          engine: VitsEngine, out_dir: Path, max_chars: int,
                          gap_sentence: float, gap_para: float,
                          gap_heading: float, pause_flag: Path | None,
                          deadline: float | None, no_resume: bool,
                          no_mp3: bool = False, log=print) -> ChapterTTSResult:
    """Synthesize a paragraph list (EPUB chapter or plain .txt) -> audio file.

    ``tag`` names the outputs: <out>/audio/<tag>/ (chunk WAVs + manifest) and
    <out>/audio/<tag>.mp3 (or .wav with no_mp3). Works for EPUB chapters and
    for `python -m tts text` alike.
    """
    t0 = time.time()
    vits_dir = Path(engine.vits_dir)
    chapter_dir = out_dir / "audio" / tag
    chapter_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    chunks = build_chunks(paragraphs, vits_dir, max_chars, gap_sentence,
                          gap_para, gap_heading, warnings)

    resumed = 0 if no_resume else resume_chunks(chapter_dir, chunks)
    manifest = {"v": MANIFEST_VERSION, "chunks": {},
                "source": source_hash(paragraphs)}
    if not no_resume:
        old = load_manifest(chapter_dir)
        old["source"] = manifest["source"]  # refresh even when resuming
        manifest = old
    todo = [c for c in chunks if c.status != "done"]
    log(f"  [{tag}] {title}: "
        f"{len(paragraphs)} paragraphs -> {len(chunks)} chunks "
        f"({resumed} cached, {len(todo)} to synthesize)")

    paused = False
    sr = engine.sample_rate
    for n, c in enumerate(todo):
        if pause_flag is not None and pause_flag.exists():
            paused = True
            break
        if deadline is not None and time.time() > deadline:
            paused = True
            break
        try:
            audio = engine.synthesize(c.text)
            write_wav(chapter_dir / c.wav_name, audio, sr)
            c.status = "done"
            c.file = c.wav_name
            c.audio_seconds = len(audio) / sr
        except KeyboardInterrupt:
            paused = True
            break                    # progress already checkpointed chunk-by-chunk
        except Exception as e:  # noqa: BLE001 - one bad chunk must not kill the run
            c.status = "error"
            c.error = str(e)[:300]
            log(f"    chunk {c.i} ERROR: {e}")
        manifest["chunks"][str(c.i)] = {
            "hash": c.hash, "file": c.file, "status": c.status,
            "audio_seconds": round(c.audio_seconds, 3),
            "gap": round(c.gap_before, 3),
            "text": c.text,
        }
        save_manifest(chapter_dir, manifest)
        if (n + 1) % 10 == 0 or n + 1 == len(todo):
            el = time.time() - t0
            rate = (n + 1) / el if el else 0
            eta = (len(todo) - n - 1) / rate if rate else 0
            log(f"    {n + 1}/{len(todo)} chunks {el:.0f}s elapsed, "
                f"ETA {eta:.0f}s")

    if pause_flag is not None and pause_flag.exists():
        pause_flag.unlink(missing_ok=True)

    # ---------- assemble audio (complete or partial) ----------
    suffix = "_partial" if paused else ""
    out_wav = chapter_dir.parent / f"{tag}{suffix}.wav"
    audio_seconds, used = assemble(chapter_dir, chunks, sr, out_wav)
    out_final = ""
    if used:
        if no_mp3:
            out_final = str(out_wav)
        else:
            out_mp3 = out_wav.with_suffix(".mp3")
            converted = False
            if out_wav.exists():
                try:
                    converted = to_mp3(out_wav, out_mp3)
                except (subprocess.CalledProcessError, OSError) as e:
                    log(f"    ffmpeg failed ({e}); keeping WAV")
            if converted:
                out_final = str(out_mp3)
            elif out_wav.exists():
                out_final = str(out_wav)
                if not shutil.which("ffmpeg"):
                    log("    ffmpeg not found - kept WAV instead of MP3")
        if not paused:
            # remove stale partial output from an earlier paused session
            (chapter_dir.parent / f"{tag}_partial.wav").unlink(missing_ok=True)
            (chapter_dir.parent / f"{tag}_partial.mp3").unlink(missing_ok=True)

    elapsed = time.time() - t0
    rtf = (audio_seconds / elapsed) if elapsed > 0 else 0.0
    return ChapterTTSResult(
        index=0, title=title, chunks=len(chunks),
        synthesized=len(todo) - sum(1 for c in todo if c.status != "done"),
        resumed=resumed, skipped_english=0,
        seconds=elapsed, audio_seconds=audio_seconds,
        realtime_factor=rtf, output=out_final, paused=paused,
        warnings=warnings[:20],
    )


def process_chapter(book: EpubBook, chapter: Chapter, engine: VitsEngine,
                    out_dir: Path, max_chars: int, gap_sentence: float,
                    gap_para: float, gap_heading: float,
                    pause_flag: Path | None, deadline: float | None,
                    no_resume: bool, no_mp3: bool = False,
                    log=print) -> ChapterTTSResult:
    paragraphs, stats = extract_paragraphs(book.path, chapter)
    if stats["skipped_english"]:
        log(f"  [{chapter.file_index:02d}] WARNING: "
            f"{stats['skipped_english']} paragraph(s) skipped - not translated "
            f"to Nepali yet")
    tag = f"{Path(book.path).stem}__ch{chapter.file_index:02d}"
    res = synthesize_paragraphs(
        tag, chapter.title or chapter.href, paragraphs, engine, out_dir,
        max_chars, gap_sentence, gap_para, gap_heading, pause_flag, deadline,
        no_resume, no_mp3, log=log)
    res.index = chapter.file_index
    res.skipped_english = stats["skipped_english"]
    return res


def concat_chapter_mp3s(out_dir: Path, stem: str, min_count: int = 2) -> Path | None:
    """Merge all complete chapter MP3s of this book into <stem>_ne_full.mp3."""
    audio_dir = out_dir / "audio"
    if not audio_dir.is_dir():
        return None
    files = sorted(f for f in audio_dir.glob(f"{stem}__ch*.mp3")
                   if "_partial" not in f.name)
    if len(files) < min_count:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    listfile = audio_dir / "concat_list.txt"
    listfile.write_text(
        "".join(f"file '{f.name}'\n" for f in files), encoding="utf-8")
    out = (out_dir / f"{stem}_full.mp3").resolve()
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(listfile), "-c:a", "libmp3lame", "-b:a", "96k",
             "-ac", "1", str(out)],
            check=True, cwd=str(audio_dir))
    except (subprocess.CalledProcessError, OSError) as e:
        print(f"note: full-book concat failed ({e})")
        return None
    finally:
        listfile.unlink(missing_ok=True)
    return out
