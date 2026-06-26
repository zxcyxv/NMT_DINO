"""Post-processing utilities for generated Old Assyrian translations."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

from .preprocessing import GAP_UNIFIED_RE, canon_decimal, frac_repl

_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?\.?\s*[^)]*\)",
    re.I,
)
_BARE_GRAM_RE = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE = re.compile(r"\(\?\)")
_CURLY_QUOT_RE = re.compile("[\u201c\u201d\u2018\u2019]")
_MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
}
_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPC_RE = re.compile(r"\s+([.,:])")
_FORBIDDEN_TRANS = str.maketrans("", "", "()——<>⌈⌋⌊[]+ʾ;")
_COMMODITY_RE = re.compile(r"-(gold|tax|textiles)\b")
_COMMODITY_REPL = {
    "gold": "pašallum gold",
    "tax": "šadduātum tax",
    "textiles": "kutānum textiles",
}
_SHEKEL_REPLS = [
    (re.compile(r"5\s+11\s*/\s*12\s+shekels?", re.I), "6 shekels less 15 grains"),
    (re.compile(r"5\s*/\s*12\s+shekels?", re.I), "⅔ shekel 15 grains"),
    (re.compile(r"7\s*/\s*12\s+shekels?", re.I), "½ shekel 15 grains"),
    (re.compile(r"1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?", re.I), "15 grains"),
]
_SLASH_ALT_RE = re.compile(r"(?<!\d)\s*/\s*(?!\d)\S+")
_STRAY_MARKS_RE = re.compile(r"<<[^>]*>>|<(?!gap\b)[^>]*>")
_MULTI_GAP_RE = re.compile(r"(?:<gap>\s*){2,}")
_PN_RE = re.compile(r"\bPN\b")
_WS_RE = re.compile(r"\s+")


def _month_repl(match: re.Match) -> str:
    return f"Month {_ROMAN2INT.get(match.group(1).upper(), match.group(1))}"


def _commodity_repl(match: re.Match) -> str:
    return _COMMODITY_REPL[match.group(1)]


class VectorizedPostprocessor:
    """Clean generated translations before MBR selection or submission."""

    def postprocess_batch(self, translations: Iterable[str]) -> list[str]:
        ser = pd.Series(list(translations)).fillna("").astype(str)
        ser = ser.str.replace(GAP_UNIFIED_RE, "<gap>", regex=True)
        ser = ser.str.replace(_PN_RE, "<gap>", regex=True)
        ser = ser.str.replace(_COMMODITY_RE, _commodity_repl, regex=True)
        for pattern, repl in _SHEKEL_REPLS:
            ser = ser.str.replace(pattern, repl, regex=True)
        ser = ser.str.replace(_EXACT_FRAC_RE, frac_repl, regex=True)
        ser = ser.str.replace(_FLOAT_RE, lambda m: canon_decimal(float(m.group(1))), regex=True)
        ser = ser.str.replace(_SOFT_GRAM_RE, " ", regex=True)
        ser = ser.str.replace(_BARE_GRAM_RE, " ", regex=True)
        ser = ser.str.replace(_UNCERTAIN_RE, "", regex=True)
        ser = ser.str.replace(_STRAY_MARKS_RE, "", regex=True)
        ser = ser.str.replace(_SLASH_ALT_RE, "", regex=True)
        ser = ser.str.replace(_CURLY_QUOT_RE, "", regex=True)
        ser = ser.str.replace(_MONTH_RE, _month_repl, regex=True)
        ser = ser.str.replace(_MULTI_GAP_RE, "<gap>", regex=True)
        ser = ser.str.replace("<gap>", "\x00GAP\x00", regex=False)
        ser = ser.str.translate(_FORBIDDEN_TRANS)
        ser = ser.str.replace("\x00GAP\x00", " <gap> ", regex=False)
        ser = ser.str.replace(_REPEAT_WORD_RE, r"\1", regex=True)
        for n in range(4, 1, -1):
            pattern = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
            ser = ser.str.replace(pattern, r"\1", regex=True)
        ser = ser.str.replace(_PUNCT_SPC_RE, r"\1", regex=True)
        ser = ser.str.replace(_REPEAT_PUNCT_RE, r"\1", regex=True)
        ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
        return ser.tolist()

