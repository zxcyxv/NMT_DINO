#!/usr/bin/env python3
"""
Evaluate DINO-pretrained vs Original ByT5 Akkadian model.
Computes BLEU and chrF++ on train.csv (labeled data).

Usage:
  python evaluate.py --original_path byt5-akkadian-optimized-34x \
                     --dino_path dino_output/final \
                     --data_path dataset/train.csv
"""

import os
import re
import math
import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

# ── sacrebleu ────────────────────────────────────────────────
try:
    import sacrebleu
except ImportError:
    print("Installing sacrebleu...")
    os.system("pip install sacrebleu")
    import sacrebleu


# ══════════════════════════════════════════════════════════════
# Preprocessing (from baseline cell 12)
# ══════════════════════════════════════════════════════════════

_V2 = re.compile(r"([aAeEiIuU])(?:2|₂)")
_V3 = re.compile(r"([aAeEiIuU])(?:3|₃)")
_ACUTE = str.maketrans({"a": "á", "e": "é", "i": "í", "u": "ú",
                         "A": "Á", "E": "É", "I": "Í", "U": "Ú"})
_GRAVE = str.maketrans({"a": "à", "e": "è", "i": "ì", "u": "ù",
                         "A": "À", "E": "È", "I": "Ì", "U": "Ù"})


def _ascii_to_diacritics(s: str) -> str:
    s = s.replace("sz", "š").replace("SZ", "Š")
    s = s.replace("s,", "ṣ").replace("S,", "Ṣ")
    s = s.replace("t,", "ṭ").replace("T,", "Ṭ")
    s = _V2.sub(lambda m: m.group(1).translate(_ACUTE), s)
    s = _V3.sub(lambda m: m.group(1).translate(_GRAVE), s)
    return s


_ALLOWED_FRACS = [
    (1/6, "0.16666"), (1/4, "0.25"), (1/3, "0.33333"),
    (1/2, "0.5"), (2/3, "0.66666"), (3/4, "0.75"), (5/6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")


def _canon_decimal(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    best = min(_ALLOWED_FRACS, key=lambda t: abs(frac - t[0]))
    if abs(frac - best[0]) <= _FRAC_TOL:
        dec = best[1]
        if ip == 0:
            return dec
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"
    return f"{x:.5f}".rstrip("0").rstrip(".")


_GAP_UNIFIED_RE = re.compile(
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
    re.I
)


def _normalize_gaps_vec(ser: pd.Series) -> pd.Series:
    return ser.str.replace(_GAP_UNIFIED_RE, "<gap>", regex=True)


_CHAR_TRANS = str.maketrans({
    "ḫ": "h", "Ḫ": "H", "ʾ": "",
    "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
    "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    "—": "-", "–": "-",
})
_SUB_X = "ₓ"

_UNICODE_UPPER = r"A-ZŠṬṢḪ\u00C0-\u00D6\u00D8-\u00DE\u0160\u1E00-\u1EFF"
_UNICODE_LOWER = r"a-zšṭṣḫ\u00E0-\u00F6\u00F8-\u00FF\u0161\u1E01-\u1EFF"
_DET_UPPER_RE = re.compile(r"\(([" + _UNICODE_UPPER + r"0-9]{1,6})\)")
_DET_LOWER_RE = re.compile(r"\(([" + _UNICODE_LOWER + r"]{1,4})\)")
_KUBABBAR_RE = re.compile(r"KÙ\.B\.")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚", "0.6666": "⅔", "0.3333": "⅓", "0.1666": "⅙",
    "0.625": "⅝", "0.75": "¾", "0.25": "¼", "0.5": "½",
}
_WS_RE = re.compile(r"\s+")


def _frac_repl(m: re.Match) -> str:
    return _EXACT_FRAC_MAP[m.group(0)]


class OptimizedPreprocessor:
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        ser = pd.Series(texts).fillna("").astype(str)
        ser = ser.apply(_ascii_to_diacritics)
        ser = ser.str.replace(_DET_UPPER_RE, r"\1", regex=True)
        ser = ser.str.replace(_DET_LOWER_RE, r"{\1}", regex=True)
        ser = _normalize_gaps_vec(ser)
        ser = ser.str.translate(_CHAR_TRANS)
        ser = ser.str.replace(_SUB_X, "", regex=False)
        ser = ser.str.replace(_KUBABBAR_RE, "KÙ.BABBAR", regex=True)
        ser = ser.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        ser = ser.str.replace(_FLOAT_RE,
                              lambda m: _canon_decimal(float(m.group(1))), regex=True)
        ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
        return ser.tolist()


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: OptimizedPreprocessor):
        proc = preprocessor.preprocess_batch(df["transliteration"].tolist())
        self.sources = ["translate Akkadian to English: " + t for t in proc]
        self.references = df["translation"].tolist()

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.sources[idx], self.references[idx]


# ══════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate_model(
    model_path: str,
    dataset: TranslationDataset,
    label: str,
    batch_size: int = 4,
    max_input_length: int = 512,
    max_new_tokens: int = 384,
    num_beams: int = 4,
    device: str = "cuda",
) -> dict:
    """Generate translations and compute BLEU / chrF++."""

    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"  Model path: {model_path}")
    print(f"{'='*60}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    if device == "cuda":
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU mem: {used:.2f} GB")

    # Use bf16 if available
    use_bf16 = (device == "cuda" and torch.cuda.is_available()
                and getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    # Generate
    all_preds = []
    all_refs = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.inference_mode():
        for batch_sources, batch_refs in tqdm(loader, desc=f"[{label}] Generating"):
            enc = tokenizer(
                list(batch_sources),
                max_length=max_input_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.inference_mode()
            with ctx:
                outputs = model.generate(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    length_penalty=1.3,
                    early_stopping=True,
                    repetition_penalty=1.2,
                    use_cache=True,
                )

            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_preds.extend(preds)
            all_refs.extend(batch_refs)

    # Cleanup
    del model
    del tokenizer
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Compute metrics
    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs])
    chrf = sacrebleu.corpus_chrf(all_preds, [all_refs], word_order=2)  # chrF++

    results = {
        "label": label,
        "model_path": model_path,
        "n_samples": len(all_preds),
        "BLEU": bleu.score,
        "chrF++": chrf.score,
        "bleu_detail": str(bleu),
        "chrf_detail": str(chrf),
        "predictions": all_preds,
        "references": all_refs,
    }

    print(f"\n  Results for [{label}]:")
    print(f"    BLEU   : {bleu.score:.2f}")
    print(f"    chrF++ : {chrf.score:.2f}")
    print(f"    {bleu}")
    print(f"    {chrf}")

    return results


def show_comparison(res_orig: dict, res_dino: dict, n_examples: int = 10):
    """Side-by-side comparison table + sample translations."""

    print(f"\n{'='*70}")
    print(f"  COMPARISON: Original vs DINO-pretrained")
    print(f"{'='*70}")
    print(f"  {'Metric':<12} {'Original':>12} {'DINO':>12} {'Delta':>12}")
    print(f"  {'-'*48}")

    bleu_delta = res_dino["BLEU"] - res_orig["BLEU"]
    chrf_delta = res_dino["chrF++"] - res_orig["chrF++"]

    bleu_sign = "+" if bleu_delta >= 0 else ""
    chrf_sign = "+" if chrf_delta >= 0 else ""

    print(f"  {'BLEU':<12} {res_orig['BLEU']:>12.2f} {res_dino['BLEU']:>12.2f} {bleu_sign}{bleu_delta:>11.2f}")
    print(f"  {'chrF++':<12} {res_orig['chrF++']:>12.2f} {res_dino['chrF++']:>12.2f} {chrf_sign}{chrf_delta:>11.2f}")
    print(f"  {'-'*48}")

    # Per-sample chrF++ differences
    metric = sacrebleu.metrics.CHRF(word_order=2)
    orig_scores = []
    dino_scores = []
    for i in range(len(res_orig["predictions"])):
        ref = res_orig["references"][i]
        s_orig = metric.sentence_score(res_orig["predictions"][i], [ref]).score
        s_dino = metric.sentence_score(res_dino["predictions"][i], [ref]).score
        orig_scores.append(s_orig)
        dino_scores.append(s_dino)

    orig_scores = np.array(orig_scores)
    dino_scores = np.array(dino_scores)
    diffs = dino_scores - orig_scores

    n_better = (diffs > 0).sum()
    n_worse = (diffs < 0).sum()
    n_same = (diffs == 0).sum()

    print(f"\n  Per-sample chrF++ comparison (N={len(diffs)}):")
    print(f"    DINO better : {n_better} ({100*n_better/len(diffs):.1f}%)")
    print(f"    DINO worse  : {n_worse} ({100*n_worse/len(diffs):.1f}%)")
    print(f"    Same        : {n_same} ({100*n_same/len(diffs):.1f}%)")
    print(f"    Mean delta  : {diffs.mean():+.2f}")
    print(f"    Median delta: {np.median(diffs):+.2f}")

    # Show examples: biggest improvements and biggest regressions
    sorted_idx = np.argsort(diffs)

    print(f"\n  {'─'*70}")
    print(f"  Top {min(n_examples, len(diffs))} DINO improvements:")
    print(f"  {'─'*70}")
    for rank, i in enumerate(reversed(sorted_idx[-n_examples:])):
        if diffs[i] <= 0:
            break
        print(f"\n  #{rank+1} | chrF++ delta: {diffs[i]:+.1f} (orig={orig_scores[i]:.1f} → dino={dino_scores[i]:.1f})")
        print(f"  REF : {res_orig['references'][i][:120]}")
        print(f"  ORIG: {res_orig['predictions'][i][:120]}")
        print(f"  DINO: {res_dino['predictions'][i][:120]}")

    print(f"\n  {'─'*70}")
    print(f"  Top {min(n_examples, len(diffs))} DINO regressions:")
    print(f"  {'─'*70}")
    for rank, i in enumerate(sorted_idx[:n_examples]):
        if diffs[i] >= 0:
            break
        print(f"\n  #{rank+1} | chrF++ delta: {diffs[i]:+.1f} (orig={orig_scores[i]:.1f} → dino={dino_scores[i]:.1f})")
        print(f"  REF : {res_orig['references'][i][:120]}")
        print(f"  ORIG: {res_orig['predictions'][i][:120]}")
        print(f"  DINO: {res_dino['predictions'][i][:120]}")

    return {
        "n_better": int(n_better),
        "n_worse": int(n_worse),
        "n_same": int(n_same),
        "mean_delta": float(diffs.mean()),
        "median_delta": float(np.median(diffs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Original vs DINO ByT5 Akkadian")
    parser.add_argument("--original_path", type=str, default="byt5-akkadian-optimized-34x",
                        help="Path to original fine-tuned model")
    parser.add_argument("--dino_path", type=str, default="dino_output/final",
                        help="Path to DINO-pretrained model checkpoint")
    parser.add_argument("--data_path", type=str, default="dataset/train.csv",
                        help="Path to labeled CSV (transliteration + translation columns)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--n_examples", type=int, default=10,
                        help="Number of example improvements/regressions to show")
    parser.add_argument("--n_samples", type=int, default=0,
                        help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--output_csv", type=str, default="eval_results.csv",
                        help="Save per-sample predictions to CSV")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print("CUDA not available, using CPU")

    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path, encoding="utf-8")
    if args.n_samples > 0:
        df = df.sample(n=min(args.n_samples, len(df)), random_state=42).reset_index(drop=True)
    print(f"  {len(df)} labeled samples")

    preprocessor = OptimizedPreprocessor()
    dataset = TranslationDataset(df, preprocessor)

    # Evaluate original model
    res_orig = evaluate_model(
        model_path=args.original_path,
        dataset=dataset,
        label="Original",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=args.device,
    )

    # Evaluate DINO model
    res_dino = evaluate_model(
        model_path=args.dino_path,
        dataset=dataset,
        label="DINO",
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=args.device,
    )

    # Comparison
    comp = show_comparison(res_orig, res_dino, n_examples=args.n_examples)

    # Save per-sample results to CSV
    out_df = pd.DataFrame({
        "reference": res_orig["references"],
        "original_pred": res_orig["predictions"],
        "dino_pred": res_dino["predictions"],
    })
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"\nPer-sample predictions saved to {args.output_csv}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  Original : BLEU={res_orig['BLEU']:.2f}  chrF++={res_orig['chrF++']:.2f}")
    print(f"  DINO     : BLEU={res_dino['BLEU']:.2f}  chrF++={res_dino['chrF++']:.2f}")
    print(f"  Delta    : BLEU={res_dino['BLEU']-res_orig['BLEU']:+.2f}  chrF++={res_dino['chrF++'] - res_orig['chrF++']:+.2f}")
    print(f"  DINO better/worse/same: {comp['n_better']}/{comp['n_worse']}/{comp['n_same']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
