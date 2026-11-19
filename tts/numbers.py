"""Nepali number -> Devanagari words (the VITS model cannot read digits).

Covers:
- 0..100 directly (standard Nepali counting table)
- larger integers with the Nepali/Indian grouping: हजार (1e3), लाख (1e5),
  करोड (1e7), अर्ब (1e9), खर्ब (1e11), सय (1e2)
- decimals: "3.14" -> तीन दशमलव एक चार (digits after the point are read out)
- thousand separators (both 1,234 and Indian 1,23,456 styles)
- percent, ranges and era abbreviations

Example: 110 -> एक सय दस
"""

from __future__ import annotations

import re

# 0..100 - standard modern Nepali forms (common spelling variants accepted)
SMALL = [
    "शुन्य",                      # 0
    "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ", "दस",      # 1-10
    "एघार", "बाह्र", "तेह्र", "चौध", "पन्ध्र", "सोह्र", "सत्र", "अठार",
    "उन्नाइस", "बीस",                                            # 11-20
    "एक्काइस", "बाइस", "तेइस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताइस",
    "अट्ठाइस", "उनन्तीस", "तीस",                                          # 21-30
    "एकतीस", "बत्तीस", "तेत्तीस", "चौँतीस", "पैंतीस", "छत्तीस", "सैँतीस",
    "अठतीस", "उनन्चालीस", "चालीस",                                          # 31-40
    "एकचालीस", "बयालीस", "त्रिचालीस", "चवालीस", "पैंतालीस", "छयालीस",
    "सतचालीस", "अठचालीस", "उनन्चास", "पचास",                                  # 41-50
    "एकाउन्न", "बाउन्न", "त्रिपन्न", "चौवन्न", "पचपन्न", "छपन्न", "सन्ताउन्न",
    "अन्ठाउन्न", "उनन्साठी", "साठी",                                          # 51-60
    "एकसठ्ठी", "बयसठ्ठी", "त्रिसठ्ठी", "चौंसठ्ठी", "पैंसठ्ठी", "छयसठ्ठी",
    "सतसठ्ठी", "अठसठ्ठी", "उनन्सत्तरी", "सत्तरी",                              # 61-70
    "एकहत्तर", "बहत्तर", "त्रिहत्तर", "चौहत्तर", "पचहत्तर", "छयहत्तर", "सतहत्तर",
    "अठहत्तर", "उनासी", "असी",                                                # 71-80
    "एकासी", "बयासी", "त्रियासी", "चौरासी", "पचासी", "छयासी", "सतासी",
    "अठासी", "उनान्नब्बे", "नब्बे",                                            # 81-90
    "एकान्नब्बे", "बयान्नब्बे", "त्रियान्नब्बे", "चौरान्नब्बे", "पन्चान्नब्बे",
    "छयान्नब्बे", "सन्तान्नब्बे", "अन्ठान्नब्बे", "उनान्सय", "एक सय",            # 91-100
]
assert len(SMALL) == 101, len(SMALL)

_UNITS = [(10 ** 11, "खर्ब"), (10 ** 9, "अर्ब"), (10 ** 7, "करोड"),
          (10 ** 5, "लाख"), (10 ** 3, "हजार")]

_DEV2LAT = str.maketrans("०१२३४५६७८९", "0123456789")

# ranges: "1990-1995" -> "1990 देखि 1995"
_RANGE_RE = re.compile(r"(\d)\s*[-\u2010-\u2015]\s*(\d)")
# a number with optional thousands separators and decimals
_NUM_RE = re.compile(r"(?:\d{1,3}(?:,\d{2,})+|\d+)(?:\.\d+)?")
# ordinal suffixes: 21st -> 21 (cardinal; Nepali ordinals need morphology)
_ORD_RE = re.compile(r"(\d)(?:st|nd|rd|th)\b", re.I)
# era abbreviations (no trailing \b: "B.C." would otherwise leave a stray dot)
_ERA = [
    (re.compile(r"\bB\.?\s*C\.?\s*E?\.?(?![A-Za-z])", re.I), " ईसा पूर्व "),
    (re.compile(r"\bA\.?\s*D\.?(?![A-Za-z])", re.I), " ईसा सन "),
]


def int_to_words(n: int) -> str:
    """Integer -> Nepali words, e.g. 110 -> 'एक सय दस', 2026 -> 'दुई हजार छब्बीस'."""
    if n < 0:
        return "माइनस " + int_to_words(-n)
    if n <= 100:
        return SMALL[n]
    parts: list[str] = []
    for value, unit in _UNITS:
        count = n // value
        if count:
            parts.append((SMALL[count] if count <= 100 else int_to_words(count)) + " " + unit)
            n -= count * value
    if n:
        hundreds, rest = divmod(n, 100)
        if hundreds:
            parts.append(SMALL[hundreds] + " सय")
        if rest:
            parts.append(SMALL[rest])
    return " ".join(parts) if parts else SMALL[0]


def number_to_words(token: str) -> str:
    """'1,23,456' / '110' / '3.14' -> Nepali words."""
    token = token.replace(",", "")
    if "." in token:
        whole, frac = token.split(".", 1)
        out = int_to_words(int(whole or "0")) + " दशमलव"
        if frac:
            out += " " + " ".join(SMALL[int(d)] for d in frac if d.isdigit())
        return out
    return int_to_words(int(token))


def spell_out(text: str) -> str:
    """Replace every number (Latin or Devanagari digits) with Nepali words.

    Also handles %, number ranges and B.C./A.D. Run this BEFORE chunking so
    sentence splitting still sees the original punctuation.
    """
    text = text.translate(_DEV2LAT)          # ११० -> 110
    for rx, repl in _ERA:                    # B.C. -> ईसा पूर्व
        text = rx.sub(repl, text)
    text = _ORD_RE.sub(r"\1", text)          # 21st -> 21
    text = _RANGE_RE.sub(r"\1 देखि \2", text)  # 1990-95 -> 1990 देखि 95
    text = text.replace("%", " प्रतिशत ")
    return _NUM_RE.sub(lambda m: number_to_words(m.group(0)), text)
