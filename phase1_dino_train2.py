#!/usr/bin/env python3
"""
Phase 1: DINO Self-Supervised Pretraining for ByT5 Akkadian
============================================================
Self-distillation (DINO) on unlabeled Akkadian transliterations from published_texts.csv.
Improves encoder representations for downstream translation.

Architecture (Length-Preserving):
  - Length-preserving corruption: ~15% of bytes → random bytes (L = L')
  - Teacher (EMA, eval): clean input → encoder → proj head → p_T  [B, L, K]
  - Student (trainable):  corrupted input → encoder → proj head → p_S [B, L, K]
  - Token-wise DINO loss: -mean(sum(p_T · log(p_S)))
  - CE loss: Student model(corrupted → reconstruct clean) via decoder
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
import time
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
    model_path: str = "byt5-akkadian-optimized-34x"
    data_path: str = "dataset/published_texts.csv"
    output_dir: str = "dino_output"

    # Model dimensions
    d_model: int = 1536
    proj_hidden: int = 3072
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

    # Length-preserving corruption
    mask_ratio: float = 0.15  # fraction of bytes to replace with random bytes

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
# Step 4: Length-Preserving Corruption
# ──────────────────────────────────────────────────────────────

def length_preserving_corruption(
    input_ids: torch.Tensor,
    mask_ratio: float = 0.15,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Replace mask_ratio fraction of content bytes with random bytes.
    Sequence length is preserved (L = L'), enabling token-wise DINO loss.

    Args:
        input_ids: [B, L] token IDs (ByT5 bytes 3-258, special 0-2, sentinel 259+)
    Returns:
        corrupted_ids: [B, L] same length, ~15% bytes replaced
        corruption_mask: [B, L] bool, True at corrupted positions
    """
    B, L = input_ids.shape
    device = input_ids.device

    corrupted = input_ids.clone()
    corruption_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

    for b in range(B):
        # Content = not pad, not eos
        content_mask = (input_ids[b] != pad_token_id) & (input_ids[b] != eos_token_id)
        content_indices = content_mask.nonzero(as_tuple=True)[0]
        n_content = content_indices.numel()

        if n_content < 2:
            continue

        # Number of tokens to corrupt
        n_corrupt = max(1, round(n_content * mask_ratio))

        # Random indices to corrupt
        perm = torch.randperm(n_content, device=device)[:n_corrupt]
        corrupt_indices = content_indices[perm]

        # Replace with random bytes (ByT5 byte range: 3-258)
        random_bytes = torch.randint(3, 259, (n_corrupt,), device=device)
        corrupted[b, corrupt_indices] = random_bytes
        corruption_mask[b, corrupt_indices] = True

    return corrupted, corruption_mask


# ──────────────────────────────────────────────────────────────
# Step 6: Projection Head + DINO Wrapper
# ──────────────────────────────────────────────────────────────

class DINOProjectionHead(nn.Module):
    """DINO projection head: Linear → GELU → LayerNorm → L2Norm → WeightNorm Linear (g=10)."""

    def __init__(self, d_model: int = 1536, hidden: int = 3072, output: int = 256,
                 initial_g: float = 10.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden)
        # Weight normalization with learnable scale g, initialized large
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(hidden, output, bias=False)
        )
        # Initialize g large so logits have meaningful magnitude from step 0
        # std(z) ≈ g * 1/√hidden → with g=10, std(z) ≈ 0.18 → std(z/τ) ≈ 4.5
        self.last_layer.weight_g.data.fill_(initial_g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.act(x)
        x = self.ln(x)
        x = F.normalize(x, dim=-1)  # L2 norm on hidden features
        x = self.last_layer(x)      # weight_norm: g * (v/||v||) · x
        return x


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

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> dict:
        """
        Length-preserving DINO forward pass.

        Teacher gets clean input, Student gets corrupted input (same length).
        DINO loss is computed token-wise on encoder outputs.
        CE loss: Student decoder reconstructs clean sequence from corrupted encoder.

        Args:
            input_ids: [B, L] raw tokenized Akkadian text
        Returns:
            dict with dino_loss, ce_loss, total_loss, diagnostics
        """
        cfg = self.cfg
        device = input_ids.device

        # ── Step 0: Length-preserving corruption ──
        corrupted_ids, corruption_mask = length_preserving_corruption(
            input_ids,
            mask_ratio=cfg.mask_ratio,
            pad_token_id=cfg.pad_token_id,
            eos_token_id=cfg.eos_token_id,
        )

        # Attention masks (same for both since length is preserved)
        attn_mask = (input_ids != cfg.pad_token_id).long()  # [B, L]
        valid_mask = attn_mask.bool()  # for loss computation

        # ── Step 1: Teacher encoder (clean input, no grad) ──
        self.teacher.eval()
        with torch.no_grad():
            teacher_enc_out = self.teacher.encoder(
                input_ids=input_ids,          # CLEAN input
                attention_mask=attn_mask,
            ).last_hidden_state               # [B, L, D]

            z_T = self.teacher_head(teacher_enc_out)  # [B, L, proj_output]

            # Centering + sharpened softmax
            z_T_centered = z_T - self.center.unsqueeze(0).unsqueeze(0)
            tau_t = getattr(self, '_current_tau_t', cfg.tau_t_start)
            p_T = F.softmax(z_T_centered / tau_t, dim=-1)  # [B, L, K]

            # Update center (excluding padding)
            if valid_mask.any():
                valid_z = z_T[valid_mask]  # [N_valid, K]
                batch_center = valid_z.mean(dim=0)
                self.center.mul_(cfg.center_momentum).add_(
                    batch_center, alpha=1.0 - cfg.center_momentum
                )

        # ── Step 2: Student encoder (corrupted input) ──
        self.student.train()
        student_enc_out = self.student.encoder(
            input_ids=corrupted_ids,          # CORRUPTED input
            attention_mask=attn_mask,
        ).last_hidden_state                   # [B, L, D] — same L!

        z_S = self.student_head(student_enc_out)  # [B, L, K]
        p_S = F.log_softmax(z_S / cfg.tau_s, dim=-1)

        # ── Step 3: Token-wise DINO loss ──
        per_token_loss = -(p_T * p_S).sum(dim=-1)  # [B, L]
        if valid_mask.any():
            dino_loss = per_token_loss[valid_mask].mean()
        else:
            dino_loss = per_token_loss.mean()

        # ── Step 4: CE loss (decoder reconstructs clean from corrupted) ──
        # labels = original clean input_ids
        # HF T5 internally: shifts labels right for decoder input, computes CE
        ce_labels = input_ids.clone()
        ce_labels[~valid_mask] = -100  # ignore padding in CE

        ce_output = self.student(
            input_ids=corrupted_ids,
            attention_mask=attn_mask,
            labels=ce_labels,
            encoder_outputs=(student_enc_out,),  # reuse cached encoder
        )
        ce_loss = ce_output.loss

        # ── Step 5: Total loss ──
        total_loss = cfg.lambda_dino * dino_loss + cfg.lambda_ce * ce_loss

        # ── Diagnostics (detached, no grad impact) ──
        with torch.no_grad():
            ln_K = math.log(cfg.proj_output)

            p_S_probs = F.softmax(z_S / cfg.tau_s, dim=-1)
            H_T = -(p_T * (p_T + 1e-10).log()).sum(dim=-1)
            H_S = -(p_S_probs * (p_S_probs + 1e-10).log()).sum(dim=-1)
            if valid_mask.any():
                H_norm_T = (H_T[valid_mask].mean() / ln_K).item()
                H_norm_S = (H_S[valid_mask].mean() / ln_K).item()
            else:
                H_norm_T = (H_T.mean() / ln_K).item()
                H_norm_S = (H_S.mean() / ln_K).item()

            z_T_valid = z_T[valid_mask] if valid_mask.any() else z_T.reshape(-1, z_T.size(-1))
            z_S_valid = z_S[valid_mask] if valid_mask.any() else z_S.reshape(-1, z_S.size(-1))

            std_z_T = z_T_valid.std().item()
            std_z_S = z_S_valid.std().item()
            abs_z_T = z_T_valid.abs().mean().item()
            abs_z_S = z_S_valid.abs().mean().item()

            center_norm = self.center.norm().item()
            mean_z_T_norm = z_T_valid.norm(dim=-1).mean().item()
            R_C = center_norm / max(mean_z_T_norm, 1e-10)

            # Intra-batch cosine similarity (per-sample mean pooled)
            B = z_T.size(0)
            sample_vecs = []
            for b in range(B):
                mb = valid_mask[b]
                if mb.any():
                    sample_vecs.append(z_T[b][mb].mean(dim=0))
            if len(sample_vecs) >= 2:
                vecs = torch.stack(sample_vecs)
                vecs_norm = F.normalize(vecs, dim=-1)
                cos_matrix = vecs_norm @ vecs_norm.T
                n = cos_matrix.size(0)
                off_diag = cos_matrix[~torch.eye(n, dtype=torch.bool, device=device)]
                S_cos = off_diag.mean().item()
            else:
                S_cos = 0.0

            # Corruption stats
            n_corrupted = corruption_mask.sum().item()
            n_total_tokens = valid_mask.sum().item()

        diagnostics = {
            "H_norm_T": H_norm_T,
            "H_norm_S": H_norm_S,
            "std_z_T": std_z_T,
            "std_z_S": std_z_S,
            "abs_z_T": abs_z_T,
            "abs_z_S": abs_z_S,
            "R_C": R_C,
            "S_cos": S_cos,
            "corrupt_ratio": n_corrupted / max(n_total_tokens, 1),
        }

        return {
            "total_loss": total_loss,
            "dino_loss": dino_loss.detach(),
            "ce_loss": ce_loss.detach(),
            "diagnostics": diagnostics,
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


def fmt_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 0:
        return "--:--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def gpu_mem_info() -> str:
    """Get GPU memory usage string."""
    if not torch.cuda.is_available():
        return "CPU"
    used = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{used:.1f}GB/{total:.0f}GB (reserved {reserved:.1f}GB)"


def train(cfg: DINOConfig):
    logger = setup_logging(cfg.output_dir)
    set_seed(cfg.seed)

    logger.info("=" * 60)
    logger.info("DINO Self-Supervised Pretraining for ByT5 Akkadian")
    logger.info("=" * 60)

    device = torch.device(cfg.device)

    # Print config summary
    logger.info(f"  Model       : {cfg.model_path}")
    logger.info(f"  Data        : {cfg.data_path}")
    logger.info(f"  Output      : {cfg.output_dir}")
    logger.info(f"  Batch       : {cfg.batch_size} x {cfg.grad_accum} accum = {cfg.batch_size * cfg.grad_accum} effective")
    logger.info(f"  Epochs      : {cfg.epochs}")
    logger.info(f"  LR          : {cfg.lr}")
    logger.info(f"  Loss weights: DINO={cfg.lambda_dino}, CE={cfg.lambda_ce}")
    logger.info(f"  EMA base    : {cfg.ema_base}")
    logger.info(f"  Temps       : student={cfg.tau_s}, teacher={cfg.tau_t_start}->{cfg.tau_t_end}")
    logger.info(f"  Device      : {device} | BF16: {cfg.use_bf16}")
    if torch.cuda.is_available():
        logger.info(f"  GPU         : {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU Memory  : {gpu_mem_info()}")
    logger.info("=" * 60)

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
    if torch.cuda.is_available():
        logger.info(f"  After model load: {gpu_mem_info()}")

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
    batches_per_epoch = len(dataloader)
    steps_per_epoch = batches_per_epoch // cfg.grad_accum
    total_steps = steps_per_epoch * cfg.epochs
    total_batches = batches_per_epoch * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info(f"  Samples     : {len(dataset)}")
    logger.info(f"  Batches/ep  : {batches_per_epoch}")
    logger.info(f"  Steps/ep    : {steps_per_epoch} (after {cfg.grad_accum}x accum)")
    logger.info(f"  Total steps : {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info("=" * 60)

    # Mixed precision
    use_amp = cfg.use_bf16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # ── Training ──
    global_step = 0
    global_batch = 0
    train_start = time.time()
    optimizer.zero_grad()

    # Running loss trackers (for smoothed display)
    running_dino = 0.0
    running_ce = 0.0
    running_total = 0.0
    running_count = 0
    prev_dino_loss = None  # For collapse detection

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        model.student.train()
        model.student_head.train()
        model.teacher.eval()
        model.teacher_head.eval()

        epoch_dino_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_total_loss = 0.0
        epoch_dino_min = float("inf")
        epoch_dino_max = float("-inf")
        num_batches = 0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            bar_format="{l_bar}{bar:20}{r_bar}",
        )
        for batch_idx, input_ids in enumerate(pbar):
            batch_start = time.time()
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
            dino_val = losses["dino_loss"].item()
            ce_val = losses["ce_loss"].item()
            total_val = losses["total_loss"].item()
            diag = losses.get("diagnostics", {})

            epoch_dino_loss += dino_val
            epoch_ce_loss += ce_val
            epoch_total_loss += total_val
            epoch_dino_min = min(epoch_dino_min, dino_val)
            epoch_dino_max = max(epoch_dino_max, dino_val)
            num_batches += 1
            global_batch += 1

            # Smoothed running averages (last ~50 batches)
            running_dino += dino_val
            running_ce += ce_val
            running_total += total_val
            running_count += 1
            if running_count > 50:
                running_dino -= running_dino / running_count
                running_ce -= running_ce / running_count
                running_total -= running_total / running_count
                running_count -= 1

            # Gradient accumulation step
            if (batch_idx + 1) % cfg.grad_accum == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.student.parameters()) + list(model.student_head.parameters()),
                    cfg.max_grad_norm,
                ).item()

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # EMA update
                ema_m = get_ema_momentum(global_step, total_steps, cfg.ema_base)
                model.update_teacher(ema_m)

                global_step += 1

            # ── Progress bar update (every batch) ──
            elapsed = time.time() - train_start
            batches_done = global_batch
            batches_left = total_batches - batches_done
            if batches_done > 0:
                secs_per_batch = elapsed / batches_done
                eta_seconds = batches_left * secs_per_batch
                samples_per_sec = (batches_done * cfg.batch_size) / elapsed
            else:
                eta_seconds = -1
                samples_per_sec = 0

            smooth_dino = running_dino / max(1, running_count)
            smooth_ce = running_ce / max(1, running_count)

            pbar.set_postfix_str(
                f"DINO={smooth_dino:.3f} CE={smooth_ce:.3f} "
                f"lr={scheduler.get_last_lr()[0]:.1e} "
                f"ETA={fmt_time(eta_seconds)} "
                f"{samples_per_sec:.1f}smp/s"
            )

            # ── Detailed logging ──
            if global_step > 0 and global_step % cfg.log_every_steps == 0 and (batch_idx + 1) % cfg.grad_accum == 0:
                avg_dino = epoch_dino_loss / num_batches
                avg_ce = epoch_ce_loss / num_batches
                avg_total = epoch_total_loss / num_batches
                lr_now = scheduler.get_last_lr()[0]
                tau_t = model._current_tau_t
                center_norm = model.center.norm().item()

                logger.info(
                    f"[Step {global_step}/{total_steps}] "
                    f"DINO={avg_dino:.4f} CE={avg_ce:.4f} Total={avg_total:.4f} | "
                    f"lr={lr_now:.2e} grad_norm={grad_norm:.2f} | "
                    f"EMA={ema_m:.4f} tau_T={tau_t:.4f} center_norm={center_norm:.3f} | "
                    f"GPU={gpu_mem_info()} | "
                    f"{samples_per_sec:.1f}smp/s ETA={fmt_time(eta_seconds)}"
                )

                # ── DINO Collapse Diagnostics ──
                if diag:
                    logger.info(
                        f"  [DIAG] H_norm: T={diag['H_norm_T']:.4f} S={diag['H_norm_S']:.4f} | "
                        f"std(z): T={diag['std_z_T']:.6f} S={diag['std_z_S']:.6f} | "
                        f"abs(z): T={diag['abs_z_T']:.6f} S={diag['abs_z_S']:.6f} | "
                        f"R_C={diag['R_C']:.4f} | S_cos={diag['S_cos']:.4f} | "
                        f"corrupt={diag.get('corrupt_ratio', 0):.1%}"
                    )
                    # Automated diagnosis
                    alerts = []
                    if diag['H_norm_T'] > 0.95:
                        alerts.append(f"H_norm(T)={diag['H_norm_T']:.3f}>0.95 → Teacher output is UNIFORM (temp/centering issue)")
                    if diag['H_norm_S'] > 0.95:
                        alerts.append(f"H_norm(S)={diag['H_norm_S']:.3f}>0.95 → Student output is UNIFORM (lr/init issue)")
                    if diag['R_C'] > 0.8:
                        alerts.append(f"R_C={diag['R_C']:.3f}>0.8 → Center is CANCELING logits (centering too aggressive)")
                    if diag['S_cos'] > 0.9:
                        alerts.append(f"S_cos={diag['S_cos']:.3f}>0.9 → MODE COLLAPSE (all samples → same vector)")
                    if diag['std_z_T'] < 0.1:
                        alerts.append(f"std(z_T)={diag['std_z_T']:.6f}<0.1 → Teacher logits have NO discrimination")
                    if diag['std_z_S'] < 0.1:
                        alerts.append(f"std(z_S)={diag['std_z_S']:.6f}<0.1 → Student logits have NO discrimination")
                    for a in alerts:
                        logger.warning(f"  >>> {a}")
                    if not alerts:
                        logger.info(f"  [DIAG] All metrics healthy")

            # Checkpoint
            if global_step > 0 and global_step % cfg.save_every_steps == 0 and (batch_idx + 1) % cfg.grad_accum == 0:
                ckpt_dir = Path(cfg.output_dir) / f"checkpoint-{global_step}"
                ckpt_dir.mkdir(exist_ok=True, parents=True)
                model.student.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                torch.save({
                    "student_head": model.student_head.state_dict(),
                    "teacher_head": model.teacher_head.state_dict(),
                    "center": model.center,
                    "global_step": global_step,
                    "epoch": epoch,
                }, ckpt_dir / "dino_state.pt")
                logger.info(f"  >> Checkpoint saved: {ckpt_dir}")

        # ── End of epoch summary ──
        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - train_start
        epochs_left = cfg.epochs - (epoch + 1)
        epoch_eta = epochs_left * epoch_elapsed

        avg_dino = epoch_dino_loss / max(1, num_batches)
        avg_ce = epoch_ce_loss / max(1, num_batches)
        avg_total = epoch_total_loss / max(1, num_batches)

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  Epoch {epoch + 1}/{cfg.epochs} complete in {fmt_time(epoch_elapsed)}")
        logger.info(f"  ─────────────────────────────────────")
        logger.info(f"  DINO loss   : {avg_dino:.4f}  (min={epoch_dino_min:.4f}, max={epoch_dino_max:.4f})")
        logger.info(f"  CE loss     : {avg_ce:.4f}")
        logger.info(f"  Total loss  : {avg_total:.4f}")
        logger.info(f"  LR          : {scheduler.get_last_lr()[0]:.2e}")
        logger.info(f"  EMA momentum: {get_ema_momentum(global_step, total_steps, cfg.ema_base):.5f}")
        logger.info(f"  Teacher temp: {get_teacher_temp(global_step, total_steps, cfg.tau_t_start, cfg.tau_t_end):.4f}")
        logger.info(f"  Center norm : {model.center.norm().item():.4f}")
        logger.info(f"  GPU memory  : {gpu_mem_info()}")
        logger.info(f"  ─────────────────────────────────────")
        logger.info(f"  Elapsed     : {fmt_time(total_elapsed)}")
        logger.info(f"  Remaining   : ~{fmt_time(epoch_eta)} ({epochs_left} epochs)")
        logger.info("=" * 60)
        logger.info("")

        # Track epoch-over-epoch DINO loss trend
        if prev_dino_loss is not None:
            delta = avg_dino - prev_dino_loss
            direction = "↓" if delta < 0 else "↑" if delta > 0 else "→"
            logger.info(f"  DINO trend  : {prev_dino_loss:.4f} {direction} {avg_dino:.4f} (delta={delta:+.4f})")
        prev_dino_loss = avg_dino

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
        logger.info(f"  >> Epoch checkpoint saved: {epoch_dir}")

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

    total_time = time.time() - train_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total time  : {fmt_time(total_time)}")
    logger.info(f"  Total steps : {global_step}")
    logger.info(f"  Final DINO  : {prev_dino_loss:.4f}")
    logger.info(f"  Checkpoint  : {final_dir}")
    logger.info(f"  Load with   : AutoModelForSeq2SeqLM.from_pretrained('{final_dir}')")
    logger.info("=" * 60)

    return model, tokenizer


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
