"""Preprocessing utilities for Old Assyrian transliteration text."""

from __future__ import annotations

import math
import re
from typing import Iterable

import pandas as pd

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({
    "a": "á", "e": "é", "i": "í", "u": "ú",
    "A": "Á", "E": "É", "I": "Í", "U": "Ú",
})
_GRAVE = str.maketrans({
    "a": "à", "e": "è", "i": "ì", "u": "ù",
    "A": "À", "E": "È", "I": "Ì", "U": "Ù",
})

_ALLOWED_FRACS = [
    (1 / 6, "0.16666"),
    (1 / 4, "0.25"),
    (1 / 3, "0.33333"),
    (1 / 2, "0.5"),
    (2 / 3, "0.66666"),
    (3 / 4, "0.75"),
    (5 / 6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")

GAP_UNIFIED_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>"
    r"|<\s*gap\s*>"
    r"|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b"
    r"|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I,
)

_CHAR_TRANS = str.maketrans({
    "ḫ": "h",
    "Ḫ": "H",
    "ʾ": "",
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "—": "-",
    "–": "-",
})
_SUB_X = "ₓ"

_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")
_KUBABBAR_RE = re.compile(r"KÙ\.B\.")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚",
    "0.6666": "⅔",
    "0.3333": "⅓",
    "0.1666": "⅙",
    "0.625": "⅝",
    "0.75": "¾",
    "0.25": "¼",
    "0.5": "½",
}
_WS_RE = re.compile(r"\s+")


def ascii_to_diacritics(text: str) -> str:
    """Convert common ASCII transliteration conventions to Unicode signs."""
    text = text.replace("sz", "š").replace("SZ", "Š")
    text = text.replace("s,", "ṣ").replace("S,", "Ṣ")
    text = text.replace("t,", "ṭ").replace("T,", "Ṭ")
    text = _V2.sub(lambda m: m.group(1).translate(_ACUTE), text)
    text = _V3.sub(lambda m: m.group(1).translate(_GRAVE), text)
    return text


def canon_decimal(value: float) -> str:
    """Map near-fraction decimals to canonical competition-style decimals."""
    integer_part = int(math.floor(value + 1e-12))
    frac = value - integer_part
    best = min(_ALLOWED_FRACS, key=lambda t: abs(frac - t[0]))
    if abs(frac - best[0]) <= _FRAC_TOL:
        dec = best[1]
        if integer_part == 0:
            return dec
        return f"{integer_part}{dec[1:]}" if dec.startswith("0.") else f"{integer_part}+{dec}"
    return f"{value:.5f}".rstrip("0").rstrip(".")


def frac_repl(match: re.Match) -> str:
    return _EXACT_FRAC_MAP[match.group(0)]


class OptimizedPreprocessor:
    """Vectorized text normalization used by training, evaluation, and submission."""

    def preprocess_batch(self, texts: Iterable[str]) -> list[str]:
        ser = pd.Series(list(texts)).fillna("").astype(str)
        ser = ser.apply(ascii_to_diacritics)
        ser = ser.str.replace(_DET_UPPER_RE, r"\1", regex=True)
        ser = ser.str.replace(_DET_LOWER_RE, r"{\1}", regex=True)
        ser = ser.str.replace(GAP_UNIFIED_RE, "<gap>", regex=True)
        ser = ser.str.translate(_CHAR_TRANS)
        ser = ser.str.replace(_SUB_X, "", regex=False)
        ser = ser.str.replace(_KUBABBAR_RE, "KÙ.BABBAR", regex=True)
        ser = ser.str.replace(_EXACT_FRAC_RE, frac_repl, regex=True)
        ser = ser.str.replace(_FLOAT_RE, lambda m: canon_decimal(float(m.group(1))), regex=True)
        ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
        return ser.tolist()

