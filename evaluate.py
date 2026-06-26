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
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

from nmt_dino.preprocessing import OptimizedPreprocessor

# ── sacrebleu ────────────────────────────────────────────────
try:
    import sacrebleu
except ImportError:
    print("Installing sacrebleu...")
    os.system("pip install sacrebleu")
    import sacrebleu


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
