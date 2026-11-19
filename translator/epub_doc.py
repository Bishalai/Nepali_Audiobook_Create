"""EPUB reading, spine listing, paragraph unit extraction and rebuilding.

Strategy: we never reconstruct XHTML by hand. Each spine document is parsed with
lxml in recover mode, only *text nodes* are swapped for their translations, and the
tree is serialized back. All markup, CSS, images and metadata stay untouched.
"""

from __future__ import annotations

import html.entities
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"

# Text is translated only when its immediate parent is one of these block tags
# (or, for tails, when the previous sibling chain is inside a body block).
BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote",
    "td", "th", "dt", "dd", "figcaption", "caption", "div", "section",
    "article", "aside", "pre", "span", "title",
}
INLINE_TAGS = {"em", "i", "b", "strong", "a", "code", "small", "sub", "sup", "u", "mark"}
SKIP_TAGS = {"script", "style", "head", "title", "metadata", "nav"}
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
NAMED_ENTITY_RE = re.compile(r"&([a-zA-Z][a-zA-Z0-9]+);")
LETTER_RE = re.compile(r"[A-Za-zÀ-ɏḀ-῿]")


@dataclass
class Unit:
    """A translatable piece of text inside one spine document."""
    file_index: int      # 1-based position in the spine (what the user selects)
    href: str            # spine href, e.g. index_split_002.html
    unit_index: int      # stable per-document index (order of appearance)
    kind: str            # "text" or "tail"
    source: str          # English text
    parent_tag: str      # immediate parent element tag
    context: str = ""    # surrounding paragraph text for inline fragments
    heading: bool = False  # inside an h1..h6


@dataclass
class Group:
    """A complete paragraph unit sent to the LLM as one context window.

    Single-sentence paragraphs and inline fragments are merged with their
    neighbours until ``chars`` reaches the target, so the model always sees a
    full, self-contained paragraph rather than an isolated sentence.
    """
    units: list[Unit]

    @property
    def chars(self) -> int:
        return sum(len(u.source) for u in self.units)

    @property
    def heading(self) -> bool:
        return bool(self.units) and all(u.heading for u in self.units)

    @property
    def text(self) -> str:
        return "\n\n".join(u.source for u in self.units)

    @property
    def unit_indexes(self) -> list[int]:
        return [u.unit_index for u in self.units]


@dataclass
class Chapter:
    file_index: int
    id: str
    href: str
    title: str
    chars: int
    unit_count: int = 0


@dataclass
class EpubBook:
    path: Path
    opf_path: str = ""
    spine: list[tuple[str, str]] = field(default_factory=list)  # (id, href)
    chapters: list[Chapter] = field(default_factory=list)


def _ent(m: re.Match) -> str:
    """Map an HTML named entity to a numeric reference (keeps the XML parser happy)."""
    name = m.group(1)
    if name in {"amp", "lt", "gt", "quot", "apos"}:
        return m.group(0)
    cp = html.entities.name2codepoint.get(name)
    return f"&#{cp};" if cp else ""


def parse_xml(data: bytes) -> etree._ElementTree:
    """Parse an XHTML document robustly (tolerates named entities / minor errors)."""
    text = data.decode("utf-8", errors="replace")
    text = NAMED_ENTITY_RE.sub(_ent, text)
    parser = etree.XMLParser(recover=True, resolve_entities=False, huge_tree=True)
    return etree.parse(io.BytesIO(text.encode("utf-8")), parser)


def _skip_zone(el: etree._Element) -> bool:
    for anc in el.iterancestors():
        if anc.tag in SKIP_TAGS if isinstance(anc.tag, str) else False:
            return True
    return False


EXTRACT_VERSION = 2  # bump whenever extraction rules change (invalidates checkpoints)


def _is_translatable(s: str) -> bool:
    """True if the node contains real words (not just punctuation/numbers).

    The threshold is deliberately low (2+ letters) so short but meaningful
    fragments like "THE", "OF", "Yes." are translated too instead of being
    left behind in English inside a Nepali paragraph.
    """
    s = s.strip()
    if len(s) < 2:
        return False
    return bool(re.search(r"[A-Za-z]{2,}", s))


def list_chapters(path: Path) -> EpubBook:
    """Open the EPUB, resolve the spine order, return chapter metadata."""
    book = EpubBook(path=path)
    with zipfile.ZipFile(path) as z:
        container = parse_xml(z.read("META-INF/container.xml"))
        opf_el = container.getroot().find(f"{{{CONTAINER_NS}}}rootfiles/{{{CONTAINER_NS}}}rootfile")
        book.opf_path = opf_el.get("full-path")
        opf_dir = str(Path(book.opf_path).parent).replace("\\", "/")
        if opf_dir == ".":
            opf_dir = ""
        opf = parse_xml(z.read(book.opf_path))
        root = opf.getroot()
        nsmap = root.nsmap or {}
        default_ns = nsmap.get(None)
        ns = {"opf": default_ns} if default_ns else {}

        manifest: dict[str, tuple[str, str]] = {}  # id -> (href, media-type)
        man = root.find("opf:manifest", ns) if ns else root.find("manifest")
        for item in man:
            manifest[item.get("id")] = (item.get("href"), item.get("media-type", ""))

        spine_el = root.find("opf:spine", ns) if ns else root.find("spine")
        for itemref in spine_el:
            iid = itemref.get("idref")
            if iid not in manifest:
                continue
            href, media = manifest[iid]
            if "html" not in media:
                continue
            full = f"{opf_dir}/{href}" if opf_dir else href
            book.spine.append((iid, full))
            book.chapters.append(Chapter(file_index=len(book.spine), id=iid, href=full,
                                         title="", chars=0))
        # titles / sizes
        for ch in book.chapters:
            data = z.read(ch.href)
            tree = parse_xml(data)
            title_el = None
            for el in tree.iter():
                if isinstance(el.tag, str) and el.tag.endswith("}title") or el.tag == "title":
                    title_el = el
                    break
            if title_el is not None and title_el.text:
                ch.title = title_el.text.strip()
            h = None
            for el in tree.iter():
                if isinstance(el.tag, str) and el.tag.split("}")[-1] in {"h1", "h2"}:
                    h = el
                    break
            body_text = "".join(t for t in tree.getroot().itertext())
            ch.chars = len(body_text)
            if h is not None:
                ht = " ".join(h.itertext()).strip()
                if ht:
                    ch.title = ht
            if not ch.title or ch.title.lower() in {"unknown", "untitled"}:
                # fall back to the first substantial heading-like line of the body
                for el in tree.iter():
                    if isinstance(el.tag, str) and el.tag.split("}")[-1] in {
                            "h1", "h2", "h3", "h4", "h5"}:
                        ht = " ".join(el.itertext()).strip()
                        if ht:
                            ch.title = ht
                            break
            ch.title = re.sub(r"\s+", " ", ch.title or "").strip()
    return book


def extract_units(path: Path, chapter: Chapter) -> list[Unit]:
    """Extract translatable text units from one spine document, in document order."""
    units: list[Unit] = []
    with zipfile.ZipFile(path) as z:
        data = z.read(chapter.href)
    tree = parse_xml(data)

    def add(el: etree._Element, kind: str, text: str) -> None:
        if not _is_translatable(text):
            return
        parent_tag = (el.tag.split("}")[-1] if isinstance(el.tag, str) else "?")
        if parent_tag in SKIP_TAGS:
            return
        if _skip_zone(el):
            return
        in_heading = parent_tag in HEADING_TAGS or any(
            isinstance(a.tag, str) and a.tag.split("}")[-1] in HEADING_TAGS
            for a in el.iterancestors())
        ctx = ""
        if parent_tag in INLINE_TAGS and not in_heading:
            anc = next((a for a in el.iterancestors()
                        if isinstance(a.tag, str) and a.tag.split("}")[-1] in BLOCK_TAGS), None)
            if anc is not None:
                ctx = " ".join(anc.itertext()).strip()[:400]
        units.append(Unit(
            file_index=chapter.file_index, href=chapter.href,
            unit_index=len(units), kind=kind, source=text.strip(),
            parent_tag=parent_tag, context=ctx, heading=in_heading,
        ))

    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        if el.text:
            add(el, "text", el.text)
        if el.tail:
            add(el, "tail", el.tail)

    chapter.unit_count = len(units)
    return units


def merge_units(units: list[Unit], target: int = 800,
                maximum: int = 1600) -> list[Group]:
    """Form complete paragraph groups for translation.

    Rules:
    - text nodes are never split, and a paragraph that on its own already meets
      ``target`` stays a standalone full paragraph;
    - consecutive short pieces (single-sentence paragraphs, salutations,
      inline fragments) are accumulated until they reach ``target`` characters,
      capped at ``maximum`` so one request stays bounded;
    - heading runs are grouped separately and collapsed into a single title
      string (a heading split into spans is one title, not fragments);
    - groups are sized adaptively by the caller from the chapter's total length.
    """
    groups: list[Group] = []
    cur: list[Unit] = []
    cur_len = 0

    def flush() -> None:
        nonlocal cur, cur_len
        if cur:
            groups.append(Group(units=cur))
            cur, cur_len = [], 0

    for u in units:
        if u.heading:
            if cur and not all(x.heading for x in cur):
                flush()
            # coalesce consecutive heading fragments into ONE title unit
            if cur and cur_len + len(u.source) > maximum:
                flush()
            cur.append(u)
            cur_len += len(u.source)
            continue
        # a heading run always ends before body text starts
        if cur and cur[-1].heading:
            flush()
        # a long paragraph must stay intact: close any pending short run first
        if cur and cur_len + len(u.source) > maximum:
            flush()
        cur.append(u)
        cur_len += len(u.source)
        if cur_len >= target:
            flush()
    flush()
    return groups


def apply_translations(path: Path, chapter: Chapter, translations: dict[int, str],
                       out_path: Path) -> None:
    """Rebuild one spine document with translated text nodes; write a full new EPUB.

    ``translations`` maps unit_index -> Nepali string. Units missing from the dict
    keep their original English text.
    """
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        blobs = {i.filename: z.read(i.filename) for i in infos}

    data = blobs[chapter.href]
    tree = parse_xml(data)
    doctype = tree.docinfo.doctype or ""

    counter = {"i": 0}

    def handle(el: etree._Element, kind: str) -> None:
        text = el.text if kind == "text" else el.tail
        if text is None or not _is_translatable(text):
            return
        parent_tag = (el.tag.split("}")[-1] if isinstance(el.tag, str) else "?")
        if parent_tag in SKIP_TAGS or _skip_zone(el):
            return
        idx = counter["i"]
        counter["i"] += 1
        if idx not in translations:
            return
        new = translations[idx]  # may be "" (collapsed heading segment)
        # preserve leading/trailing whitespace of the original node
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

    out_bytes = etree.tostring(tree, encoding="UTF-8", xml_declaration=True, doctype=doctype or None)
    blobs[chapter.href] = out_bytes
    _write_zip(path, out_path, infos, blobs)


def _write_zip(src_path: Path, out_path: Path, infos: list[zipfile.ZipInfo],
               blobs: dict[str, bytes]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w") as zout:
        for info in infos:
            zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = info.external_attr
            zout.writestr(zi, blobs[info.filename])
