"""Text preparation for Nepali VITS: normalization, symbol safety, chunking.

Pipeline per paragraph:
  spell_out(numbers) -> NFC -> sentence split (danda still intact)
  -> long-sentence split -> merge tiny fragments -> symbol safety filter

The model's alphabet is 71 symbols (Devanagari + a few ASCII punctuation
marks) and the official `nepali_cleaners` strips `! ? , । " ' ( ) ...` at
synthesis time (that is how the model was trained). We therefore:
  * split sentences BEFORE removing the danda `।`,
  * replace unsupported characters with spaces (never delete them, otherwise
    adjacent words would be glued together),
  * only warn about genuinely unexpected characters (Latin, CJK, ...).
"""

from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .numbers import spell_out

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_WS_RE = re.compile(r"\s+")

# characters the official nepali_cleaners replaces with a space
_CLEANER_STRIP = set('!?,।॥"\'()[]{}+-*/=<>@#$%^&_~`|\\')

# kept in the prepared text even though the cleaner strips them at synthesis
# time (audio is identical) - the video generator needs them to split sentences
_KEEP_BUT_STRIPPED = set("।॥")

# unicode punctuation mapped to ASCII before filtering
_PUNCT_MAP = str.maketrans({
    "\u2026": "...",            # …
    "\u2018": "'", "\u2019": "'",  # ‘ ’
    "\u201c": '"', "\u201d": '"',  # “ ”
    "\u2013": " ", "\u2014": " ",  # – —
    "\u00a0": " ",              # nbsp
    "\u200b": "",               # zero-width space
    "\u00ad": "",               # soft hyphen
})

_vits_path: Path | None = None
_allowed: set[str] | None = None


def _allowed_chars(vits_dir: Path) -> set[str]:
    global _allowed, _vits_path
    if _allowed is None:
        _vits_path = vits_dir
        if str(vits_dir) not in sys.path:
            sys.path.insert(0, str(vits_dir))
        from text.symbols import symbols  # noqa: E402 - vits package
        _allowed = set(symbols) | {"\n"}
    return _allowed


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text))


def devanagari_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    dev = sum(1 for c in letters if has_devanagari(c))
    return dev / len(letters)


@dataclass
class Chunk:
    i: int                       # global chunk index inside the chapter
    para: int                    # paragraph index (for gap decisions)
    kind: str                    # "heading" | "para"
    text: str                    # fully prepared text sent to VITS
    source: str                  # original paragraph text (for hashing)
    gap_before: float            # seconds of silence before this chunk
    hash: str = ""
    file: str = ""               # wav file name once synthesized
    status: str = "pending"      # pending | done | empty | error
    error: str = ""
    audio_seconds: float = 0.0

    def __post_init__(self) -> None:
        self.hash = hashlib.sha1(
            f"{self.text}|{self.kind}".encode("utf-8")).hexdigest()[:16]

    @property
    def wav_name(self) -> str:
        return f"c{self.i:05d}.wav"


@dataclass
class Paragraph:
    kind: str
    text: str


def normalize(raw: str) -> str:
    """Numbers -> Nepali words, NFC, unicode punctuation -> ASCII."""
    text = spell_out(raw)
    text = unicodedata.normalize("NFC", text)
    return text.translate(_PUNCT_MAP)


def safety_filter(text: str, vits_dir: Path,
                  warnings: list[str] | None = None) -> str:
    """Keep only model-supported symbols; everything else becomes a space.

    Characters the official cleaner strips anyway are replaced silently;
    anything else (Latin letters, CJK, ...) is reported as a warning.
    """
    allowed = _allowed_chars(vits_dir)
    out: list[str] = []
    unexpected: set[str] = set()
    for ch in text:
        if ch in allowed or ch in _KEEP_BUT_STRIPPED:
            out.append(ch)
        elif ch.isspace() or ch in _CLEANER_STRIP:
            out.append(" ")
        else:
            out.append(" ")
            unexpected.add(ch)
    if warnings is not None and unexpected:
        warnings.append("dropped unsupported chars: "
                        + " ".join(sorted(unexpected)[:10]))
    return _WS_RE.sub(" ", "".join(out)).strip()


def prepare(raw: str, vits_dir: Path) -> tuple[str, list[str]]:
    """numbers -> NFC -> symbol safety. Returns (clean_text, warnings)."""
    warnings: list[str] = []
    text = normalize(raw)
    latin = _LATIN_WORD_RE.findall(text)
    if latin:
        warnings.append(f"dropped Latin text: {' '.join(latin[:6])}"
                        f"{'...' if len(latin) > 6 else ''}")
    clean = safety_filter(text, vits_dir, warnings)
    return clean, warnings


def split_sentences(text: str) -> list[str]:
    """Split on Nepali/ASCII terminators, keeping closing quotes attached.

    Run this BEFORE safety_filter so the danda `।` is still present.
    """
    sentences: list[str] = []
    buf: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)
        if ch in "।!?….":
            j = i + 1
            while j < n and text[j] in "\"'”’)]":
                buf.append(text[j])
                j += 1
            if j >= n or text[j].isspace() or text[j] == "\n":
                sentences.append("".join(buf).strip())
                buf = []
                i = j
                while i < n and text[i].isspace():
                    i += 1
                continue
        i += 1
    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def split_long(sentence: str, max_chars: int) -> list[str]:
    """Break a too-long sentence at commas/full stops/word boundaries."""
    pieces: list[str] = []
    rest = sentence
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max(window.rfind("।"), window.rfind(","))
        if cut < max_chars // 3:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_chars
        pieces.append(rest[:cut + 1].strip())
        rest = rest[cut + 1:].lstrip()
    if rest.strip():
        pieces.append(rest.strip())
    return pieces


def chunk_paragraph(para_i: int, kind: str, raw: str, vits_dir: Path,
                    max_chars: int = 180, gap_sentence: float = 0.25,
                    gap_para: float = 0.65, gap_heading: float = 1.0,
                    warnings: list[str] | None = None) -> list[Chunk]:
    """Paragraph -> prepared, sentence-sized chunks with silence hints."""
    para_warn: list[str] = []
    text = normalize(raw)                       # numbers spelled out, NFC
    if not text.strip():
        return []

    # split while the danda is still there (sentence boundaries)
    parts: list[str] = []
    for sent in split_sentences(text):
        parts.extend(split_long(sent, max_chars))

    # merge tiny fragments with their neighbour for better prosody
    merged: list[str] = []
    for part in parts:
        if (merged and len(part) < 40 and len(merged[-1]) + len(part) + 1
                <= max_chars):
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part)

    chunks: list[Chunk] = []
    first_emitted = True
    for part in merged:
        clean = safety_filter(part, vits_dir, para_warn)
        if not clean:
            continue
        if first_emitted:
            gap = 0.0 if para_i == 0 else (gap_heading if kind == "heading"
                                           else gap_para)
            first_emitted = False
        else:
            gap = gap_sentence
        chunks.append(Chunk(i=-1, para=para_i, kind=kind, text=clean,
                            source=raw, gap_before=gap))
    if warnings is not None:
        latin = _LATIN_WORD_RE.findall(text)
        if latin:
            para_warn.insert(0, f"dropped Latin text: {' '.join(latin[:6])}"
                                f"{'...' if len(latin) > 6 else ''}")
        for w in para_warn:
            warnings.append(f"para {para_i}: {w}")
    return chunks
