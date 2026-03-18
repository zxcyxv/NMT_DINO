#!/usr/bin/env python3
"""
Comprehensive DINO Pipeline Diagnostic
=======================================
Checks everything: data quality, backbone health, projection head,
span corruption, and actual translation performance.

Usage:
    python diagnose.py --model_path byt5-akkadian-optimized-34x
"""

import os
import re
import math
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

# ──────────────────────────────────────────────────────────────
# Preprocessing (same as training script)
# ──────────────────────────────────────────────────────────────

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
    r"<\s*big[\s_\-]*gap\s*>|<\s*gap\s*>|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)|(?<!\w)x{2,}(?!\w)|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)|\(\s*break\s*\)|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I
)
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
_WS_RE = re.compile(r"\s+")
_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚", "0.6666": "⅔", "0.3333": "⅓", "0.1666": "⅙",
    "0.625": "⅝", "0.75": "¾", "0.25": "¼", "0.5": "½",
}

def _frac_repl(m): return _EXACT_FRAC_MAP[m.group(0)]

def preprocess_batch(texts: List[str]) -> List[str]:
    ser = pd.Series(texts).fillna("").astype(str)
    ser = ser.apply(_ascii_to_diacritics)
    ser = ser.str.replace(_DET_UPPER_RE, r"\1", regex=True)
    ser = ser.str.replace(_DET_LOWER_RE, r"{\1}", regex=True)
    ser = ser.str.replace(_GAP_UNIFIED_RE, "<gap>", regex=True)
    ser = ser.str.translate(_CHAR_TRANS)
    ser = ser.str.replace(_SUB_X, "", regex=False)
    ser = ser.str.replace(_KUBABBAR_RE, "KÙ.BABBAR", regex=True)
    ser = ser.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
    ser = ser.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)
    ser = ser.str.replace(_WS_RE, " ", regex=True).str.strip()
    return ser.tolist()


# ──────────────────────────────────────────────────────────────
# Span corruption (same as training script)
# ──────────────────────────────────────────────────────────────
import random

def byte_span_corruption(input_ids, noise_density=0.15, mean_span_len=3,
                          sentinel_start=259, eos_token_id=1, pad_token_id=0):
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    all_corrupted, all_targets = [], []

    for b in range(batch_size):
        tokens = input_ids[b].tolist()
        content_end = seq_len
        while content_end > 0 and tokens[content_end - 1] == pad_token_id:
            content_end -= 1
        if content_end > 0 and tokens[content_end - 1] == eos_token_id:
            content_end -= 1
        content = tokens[:content_end]
        content_len = len(content)

        if content_len < 4:
            all_corrupted.append(content + [eos_token_id])
            all_targets.append([sentinel_start, eos_token_id])
            continue

        num_noise_tokens = max(1, round(content_len * noise_density))
        num_spans = max(1, round(num_noise_tokens / mean_span_len))
        noise_mask = [False] * content_len
        spans_placed, attempts = 0, 0

        while spans_placed < num_spans and attempts < 100:
            span_len = max(1, int(np.random.geometric(1.0 / mean_span_len)))
            span_len = min(span_len, content_len - 1)
            start = random.randint(0, content_len - span_len)
            overlap = any(noise_mask[i] for i in range(max(0, start-1), min(content_len, start+span_len+1)))
            if not overlap:
                for i in range(start, start + span_len):
                    noise_mask[i] = True
                spans_placed += 1
            attempts += 1

        if spans_placed == 0:
            noise_mask[random.randint(0, content_len - 1)] = True

        corrupted, target = [], []
        sentinel_id = sentinel_start
        in_span = False
        for i, tok in enumerate(content):
            if noise_mask[i]:
                if not in_span:
                    corrupted.append(sentinel_id)
                    target.append(sentinel_id)
                    sentinel_id += 1
                    in_span = True
                target.append(tok)
            else:
                in_span = False
                corrupted.append(tok)
        corrupted.append(eos_token_id)
        target.append(eos_token_id)
        all_corrupted.append(corrupted)
        all_targets.append(target)

    max_c = max(len(c) for c in all_corrupted)
    max_t = max(len(t) for t in all_targets)
    cb = torch.full((batch_size, max_c), pad_token_id, dtype=torch.long, device=device)
    tb = torch.full((batch_size, max_t), pad_token_id, dtype=torch.long, device=device)
    for b in range(batch_size):
        cb[b, :len(all_corrupted[b])] = torch.tensor(all_corrupted[b], dtype=torch.long)
        tb[b, :len(all_targets[b])] = torch.tensor(all_targets[b], dtype=torch.long)
    return cb, tb


# ──────────────────────────────────────────────────────────────
# Projection Head (same as training script - fixed version)
# ──────────────────────────────────────────────────────────────

class DINOProjectionHead(nn.Module):
    def __init__(self, d_model=1536, hidden=3072, output=256):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden)
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(hidden, output, bias=False)
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.ln(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def subheader(title):
    print(f"\n  --- {title} ---")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="byt5-akkadian-optimized-34x")
    parser.add_argument("--train_path", type=str, default="dataset/train.csv")
    parser.add_argument("--published_path", type=str, default="dataset/published_texts.csv")
    parser.add_argument("--num_translate", type=int, default=20,
                        help="Number of train samples to test translation on")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ══════════════════════════════════════════════════════════
    # TEST 1: Data Quality — published_texts.csv
    # ══════════════════════════════════════════════════════════
    header("TEST 1: Published Texts Data Quality")

    pub_df = pd.read_csv(args.published_path, encoding="utf-8")
    raw_texts = pub_df["transliteration"].tolist()
    processed = preprocess_batch(raw_texts)

    print(f"  Total samples     : {len(raw_texts)}")
    print(f"  After preprocess  : {len(processed)} (non-empty: {sum(1 for t in processed if t.strip())})")

    # Length statistics
    lengths = [len(t) for t in processed if t.strip()]
    print(f"  Char lengths      : min={min(lengths)}, median={np.median(lengths):.0f}, "
          f"mean={np.mean(lengths):.0f}, max={max(lengths)}")

    # Gap-dominated samples
    gap_heavy = sum(1 for t in processed if t.count("<gap>") > len(t.split()) * 0.5)
    empty = sum(1 for t in processed if not t.strip() or t.strip() == "<gap>")
    print(f"  Empty/gap-only    : {empty}")
    print(f"  >50% gap tokens   : {gap_heavy}")

    subheader("Sample preprocessed texts (first 10)")
    for i, (raw, proc) in enumerate(zip(raw_texts[:10], processed[:10])):
        print(f"  [{i}] RAW : {str(raw)[:100]}")
        print(f"       PROC: {proc[:100]}")
        print()

    subheader("Shortest 5 samples")
    sorted_by_len = sorted(enumerate(processed), key=lambda x: len(x[1]))
    for idx, text in sorted_by_len[:5]:
        print(f"  [{idx}] ({len(text)} chars): {text[:100]}")

    subheader("Most gap-heavy 5 samples")
    gap_counts = [(i, t, t.count("<gap>")) for i, t in enumerate(processed)]
    gap_counts.sort(key=lambda x: -x[2])
    for idx, text, gc in gap_counts[:5]:
        print(f"  [{idx}] ({gc} gaps): {text[:100]}")

    # ══════════════════════════════════════════════════════════
    # TEST 2: Tokenization Sanity
    # ══════════════════════════════════════════════════════════
    header("TEST 2: Tokenization Check")

    print(f"  Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Type: {type(tokenizer).__name__}")

    # Tokenize a few samples
    sample_texts = [t for t in processed if t.strip()][:100]
    token_lengths = []
    for text in sample_texts:
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        token_lengths.append(enc.input_ids.size(1))

    print(f"  Token lengths (100 samples): min={min(token_lengths)}, "
          f"median={np.median(token_lengths):.0f}, max={max(token_lengths)}")

    # Show byte distribution for one sample
    subheader("Tokenization example")
    example_text = sample_texts[0]
    enc = tokenizer(example_text, return_tensors="pt", add_special_tokens=True)
    ids = enc.input_ids[0].tolist()
    print(f"  Text: {example_text[:80]}")
    print(f"  IDs ({len(ids)} tokens): {ids[:30]}...")
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    print(f"  Decoded back: {decoded[:80]}")

    # ══════════════════════════════════════════════════════════
    # TEST 3: Model Loading & Backbone Health
    # ══════════════════════════════════════════════════════════
    header("TEST 3: Backbone Health Check")

    print(f"  Loading model from {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
    print(f"  Config: d_model={config.d_model}, d_ff={config.d_ff}, "
          f"enc_layers={config.num_layers}, dec_layers={config.num_decoder_layers}, "
          f"vocab={config.vocab_size}")

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Check parameter statistics
    subheader("Parameter health")
    for name, param in model.named_parameters():
        if "layer.0." in name and ("weight" in name) and param.dim() >= 2:
            print(f"  {name}: shape={list(param.shape)}, "
                  f"mean={param.data.mean():.6f}, std={param.data.std():.6f}, "
                  f"abs_max={param.data.abs().max():.4f}")
        if "layer.0.layer.0" in name and "weight" in name:
            break  # Just show first layer

    # Encoder hidden state variability
    subheader("Encoder hidden state variability")
    test_texts = [t for t in processed if len(t) > 20][:8]
    enc_batch = tokenizer(test_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(device)

    with torch.no_grad():
        enc_out = model.encoder(
            input_ids=enc_batch.input_ids,
            attention_mask=enc_batch.attention_mask,
        ).last_hidden_state

    attn_mask = enc_batch.attention_mask.bool()
    valid_hidden = enc_out[attn_mask]  # [N_valid, D]

    hidden_std_per_dim = valid_hidden.std(dim=0).mean().item()
    hidden_std_per_token = valid_hidden.std(dim=-1).mean().item()
    hidden_mean_abs = valid_hidden.abs().mean().item()
    hidden_norm = valid_hidden.norm(dim=-1).mean().item()

    print(f"  std(across tokens, per dim)  : {hidden_std_per_dim:.4f}")
    print(f"  std(across dims, per token)  : {hidden_std_per_token:.4f}")
    print(f"  mean(abs(hidden))            : {hidden_mean_abs:.4f}")
    print(f"  mean(||hidden||)             : {hidden_norm:.4f}")

    if hidden_std_per_token < 0.05:
        print("  >>> WARNING: Backbone hidden states have very low variance!")
        print("  >>> The model may be improperly loaded or in a degenerate state.")
    elif hidden_std_per_token > 0.5:
        print("  >>> OK: Backbone is producing diverse hidden states.")
    else:
        print(f"  >>> Moderate variance — backbone appears functional.")

    # Inter-sample similarity
    sample_means = []
    for i in range(enc_out.size(0)):
        mask_i = attn_mask[i]
        sample_means.append(enc_out[i][mask_i].mean(dim=0))
    sample_means = torch.stack(sample_means)
    sample_means_norm = F.normalize(sample_means, dim=-1)
    cos_sim = (sample_means_norm @ sample_means_norm.T)
    n = cos_sim.size(0)
    off_diag = cos_sim[~torch.eye(n, dtype=torch.bool, device=device)]
    print(f"  Inter-sample cosine sim      : mean={off_diag.mean():.4f}, "
          f"min={off_diag.min():.4f}, max={off_diag.max():.4f}")
    if off_diag.mean() > 0.95:
        print("  >>> WARNING: All samples produce nearly identical encoder outputs!")

    # ══════════════════════════════════════════════════════════
    # TEST 4: Span Corruption Sanity Check
    # ══════════════════════════════════════════════════════════
    header("TEST 4: Span Corruption Sanity")

    test_enc = tokenizer(test_texts[:4], return_tensors="pt", padding=True,
                         truncation=True, max_length=256)
    input_ids = test_enc.input_ids.to(device)

    corrupted, targets = byte_span_corruption(input_ids)

    for i in range(min(4, corrupted.size(0))):
        orig_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        corr_ids = corrupted[i][corrupted[i] != 0].tolist()
        tgt_ids = targets[i][targets[i] != 0].tolist()
        corr_text = tokenizer.decode(corr_ids, skip_special_tokens=False)
        tgt_text = tokenizer.decode(tgt_ids, skip_special_tokens=False)

        n_sentinels_corr = sum(1 for t in corr_ids if 259 <= t <= 383)
        n_sentinels_tgt = sum(1 for t in tgt_ids if 259 <= t <= 383)

        print(f"\n  Sample {i}:")
        print(f"    Original ({len(input_ids[i][input_ids[i]!=0])} tokens): {orig_text[:80]}")
        print(f"    Corrupted ({len(corr_ids)} tokens, {n_sentinels_corr} sentinels): {corr_text[:80]}")
        print(f"    Target ({len(tgt_ids)} tokens, {n_sentinels_tgt} sentinels): {tgt_text[:80]}")

        if n_sentinels_corr != n_sentinels_tgt:
            print(f"    >>> ERROR: Sentinel count mismatch! corrupted={n_sentinels_corr}, target={n_sentinels_tgt}")
        if n_sentinels_corr == 0:
            print(f"    >>> ERROR: No spans were corrupted!")

    # ══════════════════════════════════════════════════════════
    # TEST 5: Projection Head Output Analysis
    # ══════════════════════════════════════════════════════════
    header("TEST 5: Projection Head Analysis")

    d_model = config.d_model
    proj_head = DINOProjectionHead(d_model=d_model, hidden=d_model*2, output=256).to(device)

    with torch.no_grad():
        # Feed actual encoder outputs through the head
        z = proj_head(enc_out)  # [B, L, 256]
        valid_z = z[attn_mask]  # [N_valid, 256]

    print(f"  Head output shape: {list(z.shape)}")
    print(f"  Valid tokens: {valid_z.size(0)}")
    print(f"  Logit statistics:")
    print(f"    mean      : {valid_z.mean():.6f}")
    print(f"    std       : {valid_z.std():.6f}")
    print(f"    abs mean  : {valid_z.abs().mean():.6f}")
    print(f"    min       : {valid_z.min():.6f}")
    print(f"    max       : {valid_z.max():.6f}")
    print(f"    ||z|| mean: {valid_z.norm(dim=-1).mean():.4f}")

    # Simulate softmax with different temperatures
    subheader("Softmax sharpness at different temperatures")
    for tau in [0.04, 0.07, 0.1, 0.5, 1.0]:
        p = F.softmax(valid_z / tau, dim=-1)
        entropy = -(p * (p + 1e-10).log()).sum(dim=-1)
        H_norm = entropy / math.log(256)
        max_prob = p.max(dim=-1).values

        print(f"    tau={tau:.2f}: H_norm={H_norm.mean():.4f}, "
              f"max_prob={max_prob.mean():.4f}, "
              f"entropy_range=[{H_norm.min():.4f}, {H_norm.max():.4f}]")

    # Center simulation
    subheader("Centering effect simulation")
    center = valid_z.mean(dim=0)  # [256]
    z_centered = valid_z - center
    print(f"    ||center||       : {center.norm():.4f}")
    print(f"    ||z|| mean       : {valid_z.norm(dim=-1).mean():.4f}")
    print(f"    ||z-center|| mean: {z_centered.norm(dim=-1).mean():.4f}")
    print(f"    R_C (ratio)      : {center.norm() / valid_z.norm(dim=-1).mean():.4f}")

    p_centered = F.softmax(z_centered / 0.04, dim=-1)
    H_centered = -(p_centered * (p_centered + 1e-10).log()).sum(dim=-1)
    H_norm_centered = H_centered / math.log(256)
    print(f"    After centering + tau=0.04: H_norm={H_norm_centered.mean():.4f}")

    # Inter-token cosine similarity (mode collapse check)
    subheader("Intra-batch cosine similarity (head output)")
    z_norm = F.normalize(valid_z, dim=-1)
    # Sample 200 pairs to avoid OOM
    n_valid = min(valid_z.size(0), 200)
    z_sample = z_norm[:n_valid]
    cos_mat = z_sample @ z_sample.T
    off_diag_head = cos_mat[~torch.eye(n_valid, dtype=torch.bool, device=device)[:n_valid, :n_valid]]
    print(f"    S_cos: mean={off_diag_head.mean():.4f}, "
          f"min={off_diag_head.min():.4f}, max={off_diag_head.max():.4f}")
    if off_diag_head.mean() > 0.9:
        print("    >>> WARNING: Head outputs are near-identical across tokens (mode collapse)")

    # ══════════════════════════════════════════════════════════
    # TEST 6: Actual Translation Performance
    # ══════════════════════════════════════════════════════════
    header("TEST 6: Translation Performance (Ground Truth Test)")

    train_df = pd.read_csv(args.train_path, encoding="utf-8")
    print(f"  Train samples: {len(train_df)}")

    # Sample diverse examples
    n_test = min(args.num_translate, len(train_df))
    test_indices = np.linspace(0, len(train_df)-1, n_test, dtype=int)
    test_samples = train_df.iloc[test_indices]

    raw_inputs = test_samples["transliteration"].tolist()
    references = test_samples["translation"].tolist()
    proc_inputs = preprocess_batch(raw_inputs)
    prefixed = ["translate Akkadian to English: " + t for t in proc_inputs]

    subheader(f"Translating {n_test} samples with beam search")

    predictions = []
    batch_size = 4

    for i in range(0, len(prefixed), batch_size):
        batch_texts = prefixed[i:i+batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                   enabled=(device.type == "cuda")):
            outputs = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=256,
                num_beams=4,
                length_penalty=1.3,
                repetition_penalty=1.2,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
        print(f"    Batch {i//batch_size + 1}/{(len(prefixed)+batch_size-1)//batch_size} done")

    # Show examples
    subheader("Translation examples")
    for i in range(min(10, n_test)):
        print(f"\n  [{i}] Input : {proc_inputs[i][:80]}")
        print(f"      Ref   : {references[i][:80]}")
        print(f"      Pred  : {predictions[i][:80]}")
        # Simple exact match check
        if predictions[i].strip().lower() == references[i].strip().lower():
            print(f"      >>> EXACT MATCH")

    # Compute chrF++ if sacrebleu available
    subheader("Aggregate metrics")
    try:
        import sacrebleu
        chrf = sacrebleu.metrics.CHRF(word_order=2)
        bleu = sacrebleu.metrics.BLEU()

        chrf_score = chrf.corpus_score(predictions, [references])
        bleu_score = bleu.corpus_score(predictions, [references])

        print(f"  chrF++ : {chrf_score.score:.2f}")
        print(f"  BLEU   : {bleu_score.score:.2f}")
        print(f"  (on {n_test} samples from train set)")

        if chrf_score.score < 10:
            print("  >>> CRITICAL: Model produces near-random translations!")
            print("  >>> Check: is this actually a fine-tuned translation model?")
        elif chrf_score.score < 30:
            print("  >>> WARNING: Translation quality is low.")
        else:
            print("  >>> OK: Model has meaningful translation capability.")

    except ImportError:
        print("  sacrebleu not installed — skipping chrF++/BLEU computation")
        print("  Install with: pip install sacrebleu")

        # Fallback: simple length ratio and non-empty check
        empty_preds = sum(1 for p in predictions if not p.strip())
        avg_pred_len = np.mean([len(p) for p in predictions])
        avg_ref_len = np.mean([len(r) for r in references])
        print(f"  Empty predictions: {empty_preds}/{n_test}")
        print(f"  Avg pred length: {avg_pred_len:.0f} chars")
        print(f"  Avg ref length : {avg_ref_len:.0f} chars")
        print(f"  Length ratio   : {avg_pred_len/max(avg_ref_len,1):.2f}")

    # ══════════════════════════════════════════════════════════
    # TEST 7: Span Denoising Performance (CE baseline)
    # ══════════════════════════════════════════════════════════
    header("TEST 7: Span Denoising CE (what DINO training starts from)")

    test_enc = tokenizer(test_texts[:8], return_tensors="pt", padding=True,
                         truncation=True, max_length=256).to(device)
    input_ids = test_enc.input_ids

    total_ce = 0.0
    n_batches = 0

    for _ in range(5):  # Average over 5 random corruptions
        corrupted, span_targets = byte_span_corruption(input_ids)
        corrupted_attn = (corrupted != 0).long()

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                   enabled=(device.type == "cuda")):
            out = model(
                input_ids=corrupted,
                attention_mask=corrupted_attn,
                labels=span_targets,
            )
        total_ce += out.loss.item()
        n_batches += 1

    avg_ce = total_ce / n_batches
    print(f"  Average CE loss on span denoising: {avg_ce:.4f}")
    print(f"  ln(vocab_size={config.vocab_size})     : {math.log(config.vocab_size):.4f}")
    print(f"  Ratio (CE / ln(V))               : {avg_ce / math.log(config.vocab_size):.4f}")

    if avg_ce > math.log(config.vocab_size) * 1.2:
        print("  >>> Model performs WORSE than random on denoising!")
        print("  >>> This is expected for a translation-fine-tuned model.")
        print("  >>> CE loss will be high initially during DINO training.")
    elif avg_ce < math.log(config.vocab_size) * 0.5:
        print("  >>> Model retains strong denoising capability.")
    else:
        print("  >>> Model has some denoising capability remaining.")

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    header("DIAGNOSTIC SUMMARY")
    print("  Review each section above for >>> markers indicating issues.")
    print("  Key questions answered:")
    print("    1. Data quality: Are preprocessed texts meaningful?")
    print("    2. Tokenization: Does byte encoding work correctly?")
    print("    3. Backbone: Is the pretrained model alive?")
    print("    4. Span corruption: Are corrupted/target pairs valid?")
    print("    5. Projection head: Are logit magnitudes sufficient?")
    print("    6. Translation: Does the model actually know Akkadian?")
    print("    7. Denoising: What CE loss does the model start from?")
    print()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
