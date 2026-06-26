#!/usr/bin/env python3
"""
MBR Ensemble Submission
=======================
Model A : DINO-pretrained student  (dino_ema_output/final/student)
Model B : mattiaangeli/byt5-akkadian-mbr
→ Cross-model candidate pooling + chrF++ MBR selection + post-processing
→ submission.csv
"""

import os
import gc
import re
import math
import logging
import argparse
import warnings
from pathlib import Path
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import sacrebleu

from nmt_dino.preprocessing import OptimizedPreprocessor
from nmt_dino.postprocessing import VectorizedPostprocessor

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

@dataclass
class SubmissionConfig:
    dino_model_path: str = "/kaggle/working/dino_ema_output/final/student"
    model_b_path: str    = "/kaggle/input/models/mattiaangeli/byt5-akkadian-mbr/pytorch/default/6"
    test_data_path: str  = ""   # auto-detected if empty
    lexicon_path: str    = ""   # auto-detected if empty
    output_dir: str      = "/kaggle/working"

    max_input_length: int = 512
    max_new_tokens: int   = 384
    batch_size: int       = 2
    num_workers: int      = 2
    num_buckets: int      = 6

    num_beam_cands: int       = 4
    num_beams: int            = 8
    length_penalty: float     = 1.3
    repetition_penalty: float = 1.2
    num_sample_cands: int     = 2
    mbr_top_p: float          = 0.92
    mbr_temperature: float    = 0.75
    mbr_pool_cap: int         = 32

    use_bf16: bool         = True
    use_bucket_batching: bool = True
    use_adaptive_beams: bool  = True
    checkpoint_freq: int   = 200

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        if self.device.type != "cuda":
            self.use_bf16 = False

        self.use_bf16_amp = self.use_bf16 and self._bf16_supported()

        # Auto-detect test.csv
        if not self.test_data_path:
            for root, _, files in os.walk("/kaggle/input"):
                if "test.csv" in files:
                    self.test_data_path = os.path.join(root, "test.csv")
                    break

        # Auto-detect lexicon
        if not self.lexicon_path:
            candidate = "/kaggle/input/competitions/deep-past-initiative-machine-translation/OA_Lexicon_eBL.csv"
            if os.path.exists(candidate):
                self.lexicon_path = candidate

    @staticmethod
    def _bf16_supported() -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        except Exception:
            return False


def _bf16_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


# ──────────────────────────────────────────────────────────────
# Dataset + BucketBatchSampler
# ──────────────────────────────────────────────────────────────

class AkkadianDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocessor: OptimizedPreprocessor):
        self.sample_ids = df["id"].tolist()
        proc = preprocessor.preprocess_batch(df["transliteration"].tolist())
        self.input_texts = ["translate Akkadian to English: " + t for t in proc]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.sample_ids[idx], self.input_texts[idx]


class BucketBatchSampler(Sampler):
    def __init__(self, dataset: AkkadianDataset, batch_size: int, num_buckets: int):
        lengths = [len(t.split()) for _, t in dataset]
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        bsize = max(1, len(sorted_idx) // max(1, num_buckets))
        self.buckets = [
            sorted_idx[i * bsize: None if i == num_buckets - 1 else (i + 1) * bsize]
            for i in range(num_buckets)
        ]
        self.batch_size = batch_size

    def __iter__(self):
        for bucket in self.buckets:
            for i in range(0, len(bucket), self.batch_size):
                yield bucket[i:i + self.batch_size]

    def __len__(self):
        return sum(math.ceil(len(b) / self.batch_size) for b in self.buckets)


# ──────────────────────────────────────────────────────────────
# MBR Selector
# ──────────────────────────────────────────────────────────────

def _load_lexicon(lexicon_path: str) -> Dict[str, str]:
    if not lexicon_path or not os.path.exists(lexicon_path):
        return {}
    df = pd.read_csv(lexicon_path, encoding="utf-8")
    target_types = ["PN", "GN", "DN", "RN"]
    entity_df = df[df["type"].isin(target_types)].copy()
    lexicon = {}
    for _, row in entity_df.iterrows():
        form = str(row["form"]).strip()
        norm = str(row["norm"]).strip()
        if form == "nan" or norm == "nan":
            continue
        clean = re.sub(r"[\[\]\(\)\?\!]", "", form).lower()
        if clean:
            lexicon[clean] = norm
    print(f"Loaded {len(lexicon)} proper nouns into lexicon.")
    return lexicon


class MBRSelector:
    def __init__(self, pool_cap: int = 32, lexicon: Dict[str, str] = None):
        self._metric = sacrebleu.metrics.CHRF(word_order=2)
        self.pool_cap = pool_cap
        self.lexicon = lexicon or {}
        self.w_chrf = 0.8 if self.lexicon else 1.0
        self.w_fidelity = 0.2 if self.lexicon else 0.0

    def _chrfpp(self, a: str, b: str) -> float:
        a, b = (a or "").strip(), (b or "").strip()
        if not a or not b:
            return 0.0
        return float(self._metric.sentence_score(a, [b]).score)

    def _fidelity(self, source: str, candidate: str) -> float:
        if not self.lexicon or not source or not candidate:
            return 100.0
        tokens = re.sub(r"[^\w\-\s]", "", source.lower()).split()
        expected = [self.lexicon[t].lower() for t in tokens if t in self.lexicon]
        if not expected:
            return 100.0
        cand_lower = candidate.lower()
        return (sum(1 for e in expected if e in cand_lower) / len(expected)) * 100.0

    @staticmethod
    def _dedup(xs: List[str]) -> List[str]:
        seen, out = set(), []
        for x in xs:
            x = str(x).strip()
            if x and x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def pick(self, source: str, candidates: List[str]) -> str:
        cands = self._dedup(candidates)[:self.pool_cap]
        n = len(cands)
        if n == 0:
            return ""
        if n == 1:
            return cands[0]

        best_i, best_s = 0, -1e9
        for i in range(n):
            consensus = sum(self._chrfpp(cands[i], cands[j]) for j in range(n) if j != i) / max(1, n - 1)
            fidelity = self._fidelity(source, cands[i])
            score = self.w_chrf * consensus + self.w_fidelity * fidelity
            if score > best_s:
                best_s, best_i = score, i
        return cands[best_i]


# ──────────────────────────────────────────────────────────────
# Model Wrapper
# ──────────────────────────────────────────────────────────────

class ModelWrapper:
    def __init__(self, model_path: str, cfg: SubmissionConfig, label: str):
        self.cfg = cfg
        self.label = label
        print(f"[{label}] Loading from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True).to(cfg.device).eval()
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[{label}] {n:,} parameters")

    def collate(self, batch_samples):
        ids   = [s[0] for s in batch_samples]
        texts = [s[1] for s in batch_samples]
        enc = self.tokenizer(
            texts,
            max_length=self.cfg.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return ids, enc

    def generate_candidates(self, input_ids, attention_mask) -> List[List[str]]:
        cfg = self.cfg
        B = input_ids.shape[0]
        ctx = _bf16_ctx(cfg.device, cfg.use_bf16_amp)

        if cfg.use_adaptive_beams:
            med = float(attention_mask.sum(dim=1).float().median().item())
            beam_size = cfg.num_beams if med >= 100 else max(cfg.num_beam_cands, cfg.num_beams // 2)
        else:
            beam_size = cfg.num_beams

        with ctx:
            beam_out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=beam_size,
                num_return_sequences=cfg.num_beam_cands,
                max_new_tokens=cfg.max_new_tokens,
                length_penalty=cfg.length_penalty,
                early_stopping=True,
                repetition_penalty=cfg.repetition_penalty,
                use_cache=True,
            )
            beam_texts = self.tokenizer.batch_decode(beam_out, skip_special_tokens=True)

            samp_texts = []
            if cfg.num_sample_cands > 0:
                samp_out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    num_beams=1,
                    top_p=cfg.mbr_top_p,
                    temperature=cfg.mbr_temperature,
                    num_return_sequences=cfg.num_sample_cands,
                    max_new_tokens=cfg.max_new_tokens,
                    repetition_penalty=cfg.repetition_penalty,
                    use_cache=True,
                )
                samp_texts = self.tokenizer.batch_decode(samp_out, skip_special_tokens=True)

        Rb, Rs = cfg.num_beam_cands, cfg.num_sample_cands
        pools = []
        for i in range(B):
            p = list(beam_texts[i * Rb:(i + 1) * Rb])
            if Rs > 0:
                p += list(samp_texts[i * Rs:(i + 1) * Rs])
            pools.append(p)
        return pools

    def unload(self):
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"[{self.label}] Unloaded.")


# ──────────────────────────────────────────────────────────────
# Inference + MBR
# ──────────────────────────────────────────────────────────────

def run_model(wrapper: ModelWrapper, dataset: AkkadianDataset, cfg: SubmissionConfig) -> Dict[str, List[str]]:
    if cfg.use_bucket_batching:
        sampler = BucketBatchSampler(dataset, cfg.batch_size, cfg.num_buckets)
        dl = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=cfg.num_workers,
            collate_fn=wrapper.collate,
            pin_memory=(cfg.device.type == "cuda"),
        )
    else:
        dl = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=wrapper.collate,
            pin_memory=(cfg.device.type == "cuda"),
        )

    pools_by_id = {}
    with torch.inference_mode():
        for batch_ids, enc in tqdm(dl, desc=f"[{wrapper.label}]"):
            input_ids = enc.input_ids.to(cfg.device, non_blocking=True)
            attn = enc.attention_mask.to(cfg.device, non_blocking=True)
            try:
                batch_pools = wrapper.generate_candidates(input_ids, attn)
                for sid, pool in zip(batch_ids, batch_pools):
                    pools_by_id[str(sid)] = pool
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[{wrapper.label}] OOM — skipping batch")
                    torch.cuda.empty_cache()
                    for sid in batch_ids:
                        pools_by_id.setdefault(str(sid), [])
                else:
                    raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pools_by_id


def run_submission(cfg: SubmissionConfig):
    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(cfg.output_dir) / "submission_mbr.log"),
        ],
    )
    logger = logging.getLogger("submission_mbr")

    logger.info("=" * 60)
    logger.info("MBR Ensemble Submission")
    logger.info(f"  Model A (DINO) : {cfg.dino_model_path}")
    logger.info(f"  Model B        : {cfg.model_b_path}")
    logger.info(f"  Test data      : {cfg.test_data_path}")
    logger.info(f"  BF16 AMP       : {cfg.use_bf16_amp}")
    logger.info("=" * 60)

    if not cfg.test_data_path or not os.path.exists(cfg.test_data_path):
        raise FileNotFoundError(f"test.csv not found: '{cfg.test_data_path}'")

    test_df = pd.read_csv(cfg.test_data_path, encoding="utf-8")
    logger.info(f"Test samples: {len(test_df)}")

    preprocessor = OptimizedPreprocessor()
    postprocessor = VectorizedPostprocessor()
    lexicon = _load_lexicon(cfg.lexicon_path)
    mbr = MBRSelector(pool_cap=cfg.mbr_pool_cap, lexicon=lexicon)
    dataset = AkkadianDataset(test_df, preprocessor)

    # Phase 1: Model A (DINO student)
    logger.info("Phase 1/2 — Model A (DINO student)")
    wrapper_a = ModelWrapper(cfg.dino_model_path, cfg, "DINO-student")
    pools_a = run_model(wrapper_a, dataset, cfg)
    wrapper_a.unload()
    del wrapper_a

    # Phase 2: Model B
    logger.info("Phase 2/2 — Model B")
    wrapper_b = ModelWrapper(cfg.model_b_path, cfg, "Model-B")
    pools_b = run_model(wrapper_b, dataset, cfg)
    wrapper_b.unload()
    del wrapper_b

    # Phase 3: Pool merge + MBR + postprocessing
    logger.info("Phase 3/3 — Pool merge + MBR selection")
    results: List[Tuple[str, str]] = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="MBR"):
        sid = str(row["id"])
        source = str(row["transliteration"])

        combined = pools_a.get(sid, []) + pools_b.get(sid, [])
        pp = postprocessor.postprocess_batch(combined) if combined else []
        chosen = mbr.pick(source, pp)

        if not chosen or not chosen.strip():
            chosen = "The tablet is too damaged to translate."

        results.append((sid, chosen))

        if len(results) % cfg.checkpoint_freq == 0:
            ckpt = Path(cfg.output_dir) / f"submission_ckpt_{len(results)}.csv"
            pd.DataFrame(results, columns=["id", "translation"]).to_csv(ckpt, index=False)
            logger.info(f"  Checkpoint: {len(results)} rows → {ckpt}")

    result_df = pd.DataFrame(results, columns=["id", "translation"])

    # Stats
    empty = result_df["translation"].str.strip().eq("").sum()
    lens = result_df["translation"].str.len()
    logger.info(f"Empty: {empty} | Len mean: {lens.mean():.1f} median: {lens.median():.1f}")

    out_path = Path(cfg.output_dir) / "submission.csv"
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved → {out_path} ({len(result_df)} rows)")
    print(f"\nSubmission saved: {out_path}")
    return result_df


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino_model_path", type=str, default="")
    parser.add_argument("--model_b_path",    type=str, default="")
    parser.add_argument("--test_data_path",  type=str, default="")
    parser.add_argument("--output_dir",      type=str, default="/kaggle/working")
    parser.add_argument("--batch_size",      type=int, default=2)
    parser.add_argument("--num_beams",       type=int, default=8)
    parser.add_argument("--num_beam_cands",  type=int, default=4)
    parser.add_argument("--num_sample_cands",type=int, default=2)
    parser.add_argument("--max_new_tokens",  type=int, default=384)
    parser.add_argument("--mbr_pool_cap",    type=int, default=32)
    parser.add_argument("--use_bf16",        type=lambda x: x.lower() in ("true","1"), default=True)
    args = parser.parse_args()

    cfg = SubmissionConfig(**{k: v for k, v in vars(args).items() if v})
    run_submission(cfg)
