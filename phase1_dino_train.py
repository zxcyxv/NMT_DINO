#!/usr/bin/env python3
"""
Phase 1: DINO Self-Supervised Pretraining for ByT5 Akkadian
============================================================
Self-distillation (DINO) on unlabeled Akkadian transliterations from published_texts.csv.
Improves encoder/decoder representations for downstream translation.

Architecture:
  - Shared span corruption → corrupted_input_ids + span_targets
  - Teacher (EMA, eval, no dropout): full decoder input → projection head → p_T
  - Student (trainable, train, dropout ON):
      DINO decoder: distance-masked decoder input → projection head → p_S → L_DINO
      CE decoder: encoder_outputs + labels=span_targets → L_CE
  - L_total = λ_DINO * L_DINO + λ_CE * L_CE
"""

# ──────────────────────────────────────────────────────────────
# Step 1: Config + Imports
# ──────────────────────────────────────────────────────────────

import os
import gc
import re
import math
import copy
import random
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm


@dataclass
class DINOConfig:
    # Paths
    model_path: str = "google/byt5-large"
    data_path: str = "dataset/published_texts.csv"
    output_dir: str = "dino_output"

    # Model dimensions
    d_model: int = 1024
    proj_hidden: int = 2048
    proj_output: int = 256

    # Training
    batch_size: int = 4
    grad_accum: int = 8  # effective batch size = 32
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.05

    # Loss weights
    lambda_dino: float = 1.0
    lambda_ce: float = 0.2

    # EMA / DINO
    ema_base: float = 0.996
    tau_s: float = 0.1
    tau_t_start: float = 0.04
    tau_t_end: float = 0.07
    center_momentum: float = 0.9

    # Span corruption
    noise_density: float = 0.15
    mean_span_len: int = 3
    sentinel_start: int = 259  # ByT5 sentinel tokens start

    # Distance masking
    dist_mask_pmax: float = 0.3
    dist_mask_gamma: float = 0.1

    # Tokenizer
    max_length: int = 512
    pad_token_id: int = 0
    eos_token_id: int = 1

    # Checkpointing
    save_every_steps: int = 500
    log_every_steps: int = 10
    seed: int = 42

    # Device
    device: str = "cuda"
    use_bf16: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self):
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)
        if not torch.cuda.is_available():
            self.device = "cpu"
            self.use_bf16 = False


def parse_args() -> DINOConfig:
    parser = argparse.ArgumentParser(description="DINO pretraining for ByT5 Akkadian")
    cfg = DINOConfig()
    for k, v in vars(cfg).items():
        t = type(v) if v is not None else str
        if t == bool:
            parser.add_argument(f"--{k}", type=lambda x: x.lower() in ("true", "1"), default=v)
        else:
            parser.add_argument(f"--{k}", type=t, default=v)
    args = parser.parse_args()
    return DINOConfig(**vars(args))


def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / "dino_train.log"),
        ],
    )
    return logging.getLogger("dino_train")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
# Step 2: Preprocessing (copied from baseline cell 12)
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
_WS_RE = re.compile(r"\s+")

_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚", "0.6666": "⅔", "0.3333": "⅓", "0.1666": "⅙",
    "0.625": "⅝", "0.75": "¾", "0.25": "¼", "0.5": "½",
}


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


# ──────────────────────────────────────────────────────────────
# Step 3: Dataset + DataLoader
# ──────────────────────────────────────────────────────────────

class DINOAkkadianDataset(Dataset):
    """Dataset for DINO pretraining on unlabeled Akkadian transliterations."""

    def __init__(self, csv_path: str, tokenizer, max_length: int, logger: logging.Logger):
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(csv_path, encoding="utf-8")
        raw_texts = df["transliteration"].tolist()
        logger.info(f"Loaded {len(raw_texts)} texts from {csv_path}")

        preprocessor = OptimizedPreprocessor()
        self.texts = preprocessor.preprocess_batch(raw_texts)

        # Filter empty texts
        self.texts = [t for t in self.texts if t and t.strip()]
        logger.info(f"After preprocessing/filtering: {len(self.texts)} texts")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        # Remove batch dimension: [1, L] -> [L]
        input_ids = enc.input_ids.squeeze(0)
        return input_ids


def collate_fn(batch: List[torch.Tensor], pad_token_id: int = 0) -> torch.Tensor:
    """Dynamic padding collate function."""
    max_len = max(x.size(0) for x in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :x.size(0)] = x
    return padded


# ──────────────────────────────────────────────────────────────
# Step 4: Byte-Level Span Corruption
# ──────────────────────────────────────────────────────────────

def byte_span_corruption(
    input_ids: torch.Tensor,
    noise_density: float = 0.15,
    mean_span_len: int = 3,
    sentinel_start: int = 259,
    eos_token_id: int = 1,
    pad_token_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply span corruption to a batch of byte sequences.

    Args:
        input_ids: [B, L] token IDs (bytes + special tokens)
    Returns:
        corrupted_ids: [B, L'] with sentinel tokens replacing spans
        span_targets: [B, L''] sentinel + original bytes + EOS
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    all_corrupted = []
    all_targets = []

    for b in range(batch_size):
        # Get non-padding, non-EOS tokens
        tokens = input_ids[b].tolist()

        # Find actual content (exclude pad=0 and eos=1 at end)
        content_end = seq_len
        while content_end > 0 and tokens[content_end - 1] == pad_token_id:
            content_end -= 1
        # Remove trailing EOS if present
        if content_end > 0 and tokens[content_end - 1] == eos_token_id:
            content_end -= 1

        content = tokens[:content_end]
        content_len = len(content)

        if content_len < 4:
            # Too short for corruption; use identity
            corrupted = content + [eos_token_id]
            target = [sentinel_start, eos_token_id]
            all_corrupted.append(corrupted)
            all_targets.append(target)
            continue

        # Number of tokens to mask
        num_noise_tokens = max(1, round(content_len * noise_density))
        # Number of spans
        num_spans = max(1, round(num_noise_tokens / mean_span_len))

        # Generate random span positions
        # Create a noise mask
        noise_mask = [False] * content_len

        # Place spans randomly
        spans_placed = 0
        attempts = 0
        while spans_placed < num_spans and attempts < 100:
            # Random span length (geometric-like distribution around mean)
            span_len = max(1, int(np.random.geometric(1.0 / mean_span_len)))
            span_len = min(span_len, content_len - 1)

            # Random start position
            start = random.randint(0, content_len - span_len)

            # Check if overlaps with existing span
            overlap = False
            for i in range(max(0, start - 1), min(content_len, start + span_len + 1)):
                if noise_mask[i]:
                    overlap = True
                    break

            if not overlap:
                for i in range(start, start + span_len):
                    noise_mask[i] = True
                spans_placed += 1
            attempts += 1

        if spans_placed == 0:
            # Fallback: mask a single random token
            pos = random.randint(0, content_len - 1)
            noise_mask[pos] = True

        # Build corrupted sequence and targets
        corrupted = []
        target = []
        sentinel_id = sentinel_start
        in_span = False

        for i, tok in enumerate(content):
            if noise_mask[i]:
                if not in_span:
                    # Start of a new span
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

    # Pad to same length within batch
    max_corr_len = max(len(c) for c in all_corrupted)
    max_tgt_len = max(len(t) for t in all_targets)

    corrupted_batch = torch.full((batch_size, max_corr_len), pad_token_id,
                                 dtype=torch.long, device=device)
    targets_batch = torch.full((batch_size, max_tgt_len), pad_token_id,
                               dtype=torch.long, device=device)

    for b in range(batch_size):
        c = all_corrupted[b]
        t = all_targets[b]
        corrupted_batch[b, :len(c)] = torch.tensor(c, dtype=torch.long)
        targets_batch[b, :len(t)] = torch.tensor(t, dtype=torch.long)

    return corrupted_batch, targets_batch


# ──────────────────────────────────────────────────────────────
# Step 5: Decoder Distance-Proportional Masking
# ──────────────────────────────────────────────────────────────

def distance_proportional_mask(
    decoder_input_ids: torch.Tensor,
    pad_token_id: int = 0,
    pmax: float = 0.3,
    gamma: float = 0.1,
) -> torch.Tensor:
    """
    Apply distance-proportional masking to decoder inputs (Student DINO only).
    P(mask at position t for token at relative position t-j) = min(pmax, gamma * ln(1+k))
    where k = t - j for j < t (past positions only).

    We mask each position based on its distance from the current decoding position,
    meaning earlier tokens have higher masking probability at later positions.
    Simplified: for each position t, mask with probability based on how far it is
    from the start. P(mask_t) = min(pmax, gamma * ln(1 + t)).

    Args:
        decoder_input_ids: [B, L] shifted-right span targets
    Returns:
        masked_decoder_input: [B, L] with some positions replaced by pad_token_id
    """
    B, L = decoder_input_ids.shape
    device = decoder_input_ids.device

    # Position indices: [L]
    positions = torch.arange(L, device=device, dtype=torch.float32)

    # Masking probability increases with position
    # P(mask_t) = min(pmax, gamma * ln(1 + t))
    probs = torch.clamp(gamma * torch.log1p(positions), max=pmax)

    # Don't mask position 0 (decoder start token)
    probs[0] = 0.0

    # Expand to batch: [B, L]
    probs = probs.unsqueeze(0).expand(B, -1)

    # Sample mask
    mask = torch.bernoulli(probs).bool()

    # Don't mask padding positions (they're already pad)
    is_pad = decoder_input_ids == pad_token_id
    mask = mask & ~is_pad

    # Apply mask
    masked = decoder_input_ids.clone()
    masked[mask] = pad_token_id

    return masked


# ──────────────────────────────────────────────────────────────
# Step 6: Projection Head + DINO Wrapper
# ──────────────────────────────────────────────────────────────

class DINOProjectionHead(nn.Module):
    """DINO projection head: Linear → GELU → LayerNorm → Linear → L2 Norm."""

    def __init__(self, d_model: int = 1024, hidden: int = 2048, output: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
        Returns:
            [B, L, output] L2-normalized
        """
        out = self.net(x)
        return F.normalize(out, dim=-1)


class DINOByT5(nn.Module):
    """
    DINO wrapper around ByT5 for self-supervised pretraining.

    Contains:
    - Student: trainable ByT5 + projection head
    - Teacher: EMA copy of student (no gradients)
    - Center buffer for teacher output centering
    """

    def __init__(self, cfg: DINOConfig, logger: logging.Logger):
        super().__init__()
        self.cfg = cfg
        self.logger = logger

        # Load student model
        logger.info(f"Loading student model from {cfg.model_path}")
        self.student = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_path)

        if cfg.gradient_checkpointing:
            self.student.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Create teacher as deep copy
        logger.info("Creating teacher (deep copy of student)")
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # Projection heads
        self.student_head = DINOProjectionHead(cfg.d_model, cfg.proj_hidden, cfg.proj_output)
        self.teacher_head = DINOProjectionHead(cfg.d_model, cfg.proj_hidden, cfg.proj_output)

        # Initialize teacher head from student head
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Center buffer
        self.register_buffer("center", torch.zeros(cfg.proj_output))

        n_student = sum(p.numel() for p in self.student.parameters())
        n_head = sum(p.numel() for p in self.student_head.parameters())
        logger.info(f"Student params: {n_student:,} + head: {n_head:,}")

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        """EMA update of teacher model and head."""
        for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
            t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
        for t_param, s_param in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)

    def shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        """Shift labels right for decoder input (T5 style: pad token prepended)."""
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = self.cfg.pad_token_id  # T5 uses pad as decoder start
        return shifted

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> dict:
        """
        Full DINO forward pass.

        Args:
            input_ids: [B, L] raw tokenized Akkadian text

        Returns:
            dict with dino_loss, ce_loss, total_loss
        """
        cfg = self.cfg
        device = input_ids.device

        # ── Step 0: Span corruption (shared) ──
        corrupted_ids, span_targets = byte_span_corruption(
            input_ids,
            noise_density=cfg.noise_density,
            mean_span_len=cfg.mean_span_len,
            sentinel_start=cfg.sentinel_start,
            eos_token_id=cfg.eos_token_id,
            pad_token_id=cfg.pad_token_id,
        )

        # Attention masks
        corrupted_attn = (corrupted_ids != cfg.pad_token_id).long()
        target_mask = (span_targets != cfg.pad_token_id)  # [B, L_dec] bool

        # Decoder inputs
        decoder_input_clean = self.shift_right(span_targets)  # Teacher
        decoder_input_masked = distance_proportional_mask(    # Student DINO
            decoder_input_clean,
            pad_token_id=cfg.pad_token_id,
            pmax=cfg.dist_mask_pmax,
            gamma=cfg.dist_mask_gamma,
        )

        # ── Step 1: Teacher forward (no grad, eval mode) ──
        self.teacher.eval()
        with torch.no_grad():
            teacher_enc_out = self.teacher.encoder(
                input_ids=corrupted_ids,
                attention_mask=corrupted_attn,
            ).last_hidden_state

            teacher_dec_out = self.teacher.decoder(
                input_ids=decoder_input_clean,
                encoder_hidden_states=teacher_enc_out,
                encoder_attention_mask=corrupted_attn,
            ).last_hidden_state

            # Project and compute teacher distribution
            z_T = self.teacher_head(teacher_dec_out)  # [B, L_dec, proj_output]

            # Centering + sharpened softmax
            z_T_centered = z_T - self.center.unsqueeze(0).unsqueeze(0)

            # Get current teacher temperature (will be set externally via tau_t attr)
            tau_t = getattr(self, '_current_tau_t', cfg.tau_t_start)
            p_T = F.softmax(z_T_centered / tau_t, dim=-1)  # [B, L_dec, proj_output]

            # Update center (excluding padding)
            if target_mask.any():
                valid_z = z_T[target_mask]  # [N_valid, proj_output]
                batch_center = valid_z.mean(dim=0)  # [proj_output]
                self.center.mul_(cfg.center_momentum).add_(
                    batch_center, alpha=1.0 - cfg.center_momentum
                )

        # ── Step 2: Student encoder (single pass, cached) ──
        self.student.train()
        student_enc_out = self.student.encoder(
            input_ids=corrupted_ids,
            attention_mask=corrupted_attn,
        ).last_hidden_state

        # ── Step 3: Student DINO decoder ──
        student_dec_out = self.student.decoder(
            input_ids=decoder_input_masked,
            encoder_hidden_states=student_enc_out,
            encoder_attention_mask=corrupted_attn,
            output_hidden_states=True,
        )
        # Use last hidden state
        student_hidden = student_dec_out.last_hidden_state  # [B, L_dec, d_model]
        z_S = self.student_head(student_hidden)  # [B, L_dec, proj_output]

        # Student log-softmax
        p_S = F.log_softmax(z_S / cfg.tau_s, dim=-1)  # [B, L_dec, proj_output]

        # DINO loss: cross-entropy between teacher and student distributions
        per_token_loss = -(p_T * p_S).sum(dim=-1)  # [B, L_dec]

        if target_mask.any():
            dino_loss = per_token_loss[target_mask].mean()
        else:
            dino_loss = per_token_loss.mean()

        # ── Step 4: Student CE decoder ──
        # Use HuggingFace's built-in CE loss (internally does shift-right + cross-entropy)
        ce_output = self.student(
            input_ids=corrupted_ids,
            attention_mask=corrupted_attn,
            labels=span_targets,
            encoder_outputs=(student_enc_out,),  # Reuse cached encoder output
        )
        ce_loss = ce_output.loss

        # ── Step 5: Total loss ──
        total_loss = cfg.lambda_dino * dino_loss + cfg.lambda_ce * ce_loss

        return {
            "total_loss": total_loss,
            "dino_loss": dino_loss.detach(),
            "ce_loss": ce_loss.detach(),
        }


# ──────────────────────────────────────────────────────────────
# Step 8: Training Loop
# ──────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate scheduler with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_ema_momentum(step: int, total_steps: int, base: float = 0.996) -> float:
    """Cosine annealing of EMA momentum from base to 1.0."""
    return 1.0 - (1.0 - base) * (math.cos(math.pi * step / total_steps) + 1.0) / 2.0


def get_teacher_temp(step: int, total_steps: int, t_start: float, t_end: float) -> float:
    """Linear warmup of teacher temperature over first 30% of training."""
    warmup_steps = int(0.3 * total_steps)
    if step < warmup_steps:
        return t_start + (t_end - t_start) * step / warmup_steps
    return t_end


def train(cfg: DINOConfig):
    logger = setup_logging(cfg.output_dir)
    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("DINO Self-Supervised Pretraining for ByT5 Akkadian")
    logger.info("=" * 60)
    logger.info(f"Config: {vars(cfg)}")

    device = torch.device(cfg.device)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)

    # Create dataset and dataloader
    dataset = DINOAkkadianDataset(cfg.data_path, tokenizer, cfg.max_length, logger)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda batch: collate_fn(batch, cfg.pad_token_id),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # Create model
    model = DINOByT5(cfg, logger).to(device)

    # Optimizer: only student + student_head parameters
    optimizer_params = [
        {"params": model.student.parameters(), "lr": cfg.lr},
        {"params": model.student_head.parameters(), "lr": cfg.lr},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler
    steps_per_epoch = len(dataloader) // cfg.grad_accum
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info(f"Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}, "
                f"Warmup: {warmup_steps}")

    # Mixed precision
    use_amp = cfg.use_bf16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    logger.info(f"AMP: {use_amp}, dtype: {amp_dtype}")

    # Training
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(cfg.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.epochs}")
        model.student.train()
        model.student_head.train()
        model.teacher.eval()
        model.teacher_head.eval()

        epoch_dino_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        for batch_idx, input_ids in enumerate(pbar):
            input_ids = input_ids.to(device, non_blocking=True)

            # Set current teacher temperature
            model._current_tau_t = get_teacher_temp(
                global_step, total_steps, cfg.tau_t_start, cfg.tau_t_end
            )

            # Forward pass with autocast
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                losses = model(input_ids)
                loss = losses["total_loss"] / cfg.grad_accum

            # Backward
            scaler.scale(loss).backward()

            # Accumulate metrics
            epoch_dino_loss += losses["dino_loss"].item()
            epoch_ce_loss += losses["ce_loss"].item()
            epoch_total_loss += losses["total_loss"].item()
            num_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % cfg.grad_accum == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.student.parameters()) + list(model.student_head.parameters()),
                    cfg.max_grad_norm,
                )

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # EMA update
                ema_m = get_ema_momentum(global_step, total_steps, cfg.ema_base)
                model.update_teacher(ema_m)

                global_step += 1

                # Logging
                if global_step % cfg.log_every_steps == 0:
                    avg_dino = epoch_dino_loss / num_batches
                    avg_ce = epoch_ce_loss / num_batches
                    avg_total = epoch_total_loss / num_batches
                    lr_now = scheduler.get_last_lr()[0]
                    tau_t = model._current_tau_t

                    pbar.set_postfix({
                        "step": global_step,
                        "dino": f"{avg_dino:.4f}",
                        "ce": f"{avg_ce:.4f}",
                        "lr": f"{lr_now:.2e}",
                        "ema": f"{ema_m:.4f}",
                        "τT": f"{tau_t:.4f}",
                    })
                    logger.info(
                        f"Step {global_step} | "
                        f"DINO: {avg_dino:.4f} | CE: {avg_ce:.4f} | "
                        f"Total: {avg_total:.4f} | LR: {lr_now:.2e} | "
                        f"EMA: {ema_m:.4f} | τ_T: {tau_t:.4f}"
                    )

                # Checkpoint
                if global_step % cfg.save_every_steps == 0:
                    ckpt_dir = Path(cfg.output_dir) / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(exist_ok=True, parents=True)
                    model.student.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    # Save projection heads
                    torch.save({
                        "student_head": model.student_head.state_dict(),
                        "teacher_head": model.teacher_head.state_dict(),
                        "center": model.center,
                        "global_step": global_step,
                        "epoch": epoch,
                    }, ckpt_dir / "dino_state.pt")
                    logger.info(f"Checkpoint saved: {ckpt_dir}")

        # End of epoch stats
        avg_dino = epoch_dino_loss / max(1, num_batches)
        avg_ce = epoch_ce_loss / max(1, num_batches)
        avg_total = epoch_total_loss / max(1, num_batches)
        logger.info(
            f"Epoch {epoch + 1} complete | "
            f"DINO: {avg_dino:.4f} | CE: {avg_ce:.4f} | Total: {avg_total:.4f}"
        )

        # Save epoch checkpoint
        epoch_dir = Path(cfg.output_dir) / f"epoch-{epoch + 1}"
        epoch_dir.mkdir(exist_ok=True, parents=True)
        model.student.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        torch.save({
            "student_head": model.student_head.state_dict(),
            "teacher_head": model.teacher_head.state_dict(),
            "center": model.center,
            "global_step": global_step,
            "epoch": epoch,
        }, epoch_dir / "dino_state.pt")
        logger.info(f"Epoch checkpoint saved: {epoch_dir}")

    # ── Final save (HF-compatible) ──
    final_dir = Path(cfg.output_dir) / "final"
    final_dir.mkdir(exist_ok=True, parents=True)
    model.student.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    torch.save({
        "student_head": model.student_head.state_dict(),
        "teacher_head": model.teacher_head.state_dict(),
        "center": model.center,
        "global_step": global_step,
    }, final_dir / "dino_state.pt")

    logger.info("=" * 60)
    logger.info(f"Training complete! Final checkpoint: {final_dir}")
    logger.info(f"Total steps: {global_step}")
    logger.info(f"Load with: AutoModelForSeq2SeqLM.from_pretrained('{final_dir}')")
    logger.info("=" * 60)

    return model, tokenizer


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
