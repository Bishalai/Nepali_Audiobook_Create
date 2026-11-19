"""Quick self-test for tts prep/pipeline (run: python scripts/test_tts_prep.py)."""
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from translator.epub_doc import list_chapters           # noqa: E402
from tts.pipeline import extract_paragraphs, build_chunks  # noqa: E402

book = list_chapters(Path("output/sophies_world_ne_ch4.epub"))
ch = [c for c in book.chapters if c.file_index == 4][0]
paras, stats = extract_paragraphs(book.path, ch)
warns: list[str] = []
chunks = build_chunks(paras, Path("third_party/vits"), 180, 0.25, 0.65, 1.0,
                      warns)
print("paragraphs:", len(paras), "| chunks:", len(chunks))
print("max chunk len:", max(len(c.text) for c in chunks))
print("warnings:", len(warns))
for w in warns[:8]:
    print("  W:", w)
print()
ends = collections.Counter(c.text[-1] for c in chunks)
print("last chars:", dict(ends))
for c in chunks[1:4]:
    print(f"  [{c.kind}] gap={c.gap_before} | {c.text[:110]}")

bad = [c.text for c in chunks
       if re.search(r"[A-Za-z]{2,}|\u4e4b\u82b1|\u09ac", c.text)]
print("chunks with unexpected scripts:", len(bad))
odd = [c.text for c in chunks
       if re.search(r"[^\u0900-\u097F .,:;!?()'\u201c\u201d\u2018\u2019-]",
                    c.text)]
print("chunks with out-of-alphabet chars:", len(odd))
for t in odd[:3]:
    print("  ODD:", t[:120])
