#!/usr/bin/env python3
"""
Phase 1: DINO EMA Pretraining + Sequence-Level KD for ByT5 Akkadian (Single GPU)
=================================================================================

Architecture (Length-Preserving + Asymmetric KD):
  - Length-preserving corruption: ~15% of bytes → random bytes (L = L')
  - Teacher (EMA, eval): clean input → encoder → proj head → p_T (Encoder)
                         clean input → decoder -> p_T_dec (Decoder)
  - Student (trainable): corrupted input → encoder → proj head → p_S (Encoder)
                         corrupted input → decoder -> p_S_dec (Decoder)
  - Token-wise DINO loss: -mean(sum(p_T · log(p_S))) (Encoder self-distillation)
  - Sequence-Level KD: KL-Divergence(p_T_dec || p_S_dec) (Decoder knowledge distillation)
  - Teacher is updated strictly via Exponential Moving Average (EMA).
  - L_total = λ_DINO * L_DINO + λ_KD * L_KD
"""

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
class DINOEMAConfig:
    # Paths
    model_path: str = "/kaggle/input/byt5-akkadian-optimized-34x/byt5-akkadian-optimized-34x"
    data_path: str = "/kaggle/input/deep-past-initiative-machine-translation/published_texts.csv"
    output_dir: str = "/kaggle/working/dino_ema_output"

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
    lambda_kd: float = 1.0

    # EMA / DINO
    ema_base: float = 0.996
    tau_s: float = 0.1
    tau_t_start: float = 0.04
    tau_t_end: float = 0.07
    center_momentum: float = 0.9

    # Length-preserving corruption
    mask_ratio: float = 0.15

    # Tokenizer
    max_length: int = 512
    pad_token_id: int = 0
    eos_token_id: int = 1

    # Translation sample check (labeled data)
    train_data_path: str = ""  # train.csv with transliteration/translation columns
    sample_check_every: int = 80  # steps between translation sample checks

    # Checkpointing
    save_every_steps: int = 500
    log_every_steps: int = 10
    seed: int = 42

    # Device / precision
    device: str = "cuda"
    use_bf16: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self):
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)


def parse_args() -> DINOEMAConfig:
    parser = argparse.ArgumentParser(description="DINO EMA + Seq-KD for ByT5 Akkadian (Single GPU)")
    cfg = DINOEMAConfig()
    for k, v in vars(cfg).items():
        t = type(v) if v is not None else str
        if t == bool:
            parser.add_argument(f"--{k}", type=lambda x: x.lower() in ("true", "1"), default=v)
        else:
            parser.add_argument(f"--{k}", type=t, default=v)
    args = parser.parse_args()
    return DINOEMAConfig(**vars(args))


def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(output_dir) / "dino_ema_train.log"),
        ],
    )
    return logging.getLogger("dino_ema_train")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
# Preprocessing
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
# Dataset + DataLoader
# ──────────────────────────────────────────────────────────────

class DINOAkkadianDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_length: int, logger: logging.Logger):
        df = pd.read_csv(csv_path, encoding="utf-8")
        raw_texts = df["transliteration"].tolist()
        logger.info(f"Loaded {len(raw_texts)} texts from {csv_path}")

        preprocessor = OptimizedPreprocessor()
        texts = preprocessor.preprocess_batch(raw_texts)
        texts = [t for t in texts if t and t.strip()]
        logger.info(f"After preprocessing/filtering: {len(texts)} texts")

        # [Fix #2] Pre-tokenize once at init — avoids re-tokenizing every __getitem__ call
        logger.info("Pre-tokenizing all texts...")
        self.input_ids: List[torch.Tensor] = []
        for text in texts:
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=True,
            )
            self.input_ids.append(enc.input_ids.squeeze(0))
        logger.info(f"Pre-tokenization complete: {len(self.input_ids)} samples cached")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


def collate_fn(batch: List[torch.Tensor], pad_token_id: int = 0) -> torch.Tensor:
    max_len = max(x.size(0) for x in batch)
    padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :x.size(0)] = x
    return padded


# ──────────────────────────────────────────────────────────────
# Length-Preserving Corruption
# ──────────────────────────────────────────────────────────────

def length_preserving_corruption(
    input_ids: torch.Tensor,
    mask_ratio: float = 0.15,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # [Fix #4] Fully vectorized — no Python for-loop over batch items
    B, L = input_ids.shape
    device = input_ids.device

    content_mask = (input_ids != pad_token_id) & (input_ids != eos_token_id)  # [B, L]

    # Random scores: content positions get uniform [0,1), non-content gets 2.0 (pushed to back)
    rand_scores = torch.rand(B, L, device=device)
    rand_scores[~content_mask] = 2.0

    # rank[b, l] = position of token l in ascending score order
    rank = rand_scores.argsort(dim=1).argsort(dim=1)

    n_content = content_mask.sum(dim=1)  # [B]
    n_corrupt = (n_content.float() * mask_ratio).round().clamp(min=1).long()  # [B]

    # Positions with rank < n_corrupt[b] AND content → corrupted
    corruption_mask = (rank < n_corrupt.unsqueeze(1)) & content_mask  # [B, L]

    random_bytes = torch.randint(3, 259, (B, L), device=device)
    corrupted = input_ids.clone()
    corrupted[corruption_mask] = random_bytes[corruption_mask]

    return corrupted, corruption_mask


# ──────────────────────────────────────────────────────────────
# Projection Head + DINO Wrapper
# ──────────────────────────────────────────────────────────────

class DINOProjectionHead(nn.Module):
    def __init__(self, d_model: int = 1536, hidden: int = 3072, output: int = 256,
                 initial_g: float = 10.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden)
        self.last_layer = nn.utils.weight_norm(nn.Linear(hidden, output, bias=False))
        self.last_layer.weight_g.data.fill_(initial_g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.act(x)
        x = self.ln(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


class DINOEMA(nn.Module):
    def __init__(self, cfg: DINOEMAConfig, logger: logging.Logger):
        super().__init__()
        self.cfg = cfg
        self.logger = logger
        self._current_tau_t = cfg.tau_t_start

        logger.info(f"Loading student model from {cfg.model_path}")
        self.student = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_path)

        if cfg.gradient_checkpointing:
            self.student.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        logger.info("Creating teacher (deep copy of student)")
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student_head = DINOProjectionHead(cfg.d_model, cfg.proj_hidden, cfg.proj_output)
        self.teacher_head = DINOProjectionHead(cfg.d_model, cfg.proj_hidden, cfg.proj_output)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_head.parameters():
            p.requires_grad = False

        # Teacher는 항상 eval — forward 안에서 매번 세팅할 필요 없음
        self.teacher.eval()
        self.teacher_head.eval()

        self.register_buffer("center", torch.zeros(cfg.proj_output))

        n_student = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        n_head = sum(p.numel() for p in self.student_head.parameters() if p.requires_grad)
        logger.info(f"Trainable Student params: {n_student:,} + Trainable head params: {n_head:,}")

    def train(self, mode: bool = True):
        """model.train() 호출 시 teacher는 항상 eval로 고정."""
        super().train(mode)
        self.teacher.eval()
        self.teacher_head.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        for t_p, s_p in zip(self.teacher.parameters(), self.student.parameters()):
            t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)
        for t_p, s_p in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            t_p.data.mul_(momentum).add_(s_p.data, alpha=1.0 - momentum)

    def forward(self, input_ids: torch.Tensor) -> dict:
        cfg = self.cfg

        corrupted_ids, corruption_mask = length_preserving_corruption(
            input_ids,
            mask_ratio=cfg.mask_ratio,
            pad_token_id=cfg.pad_token_id,
            eos_token_id=cfg.eos_token_id,
        )

        attn_mask = (input_ids != cfg.pad_token_id).long()
        valid_mask = attn_mask.bool()

        labels = input_ids.clone()
        labels[~valid_mask] = -100

        # [Fix #3] Pass decoder_input_ids directly instead of labels
        # — avoids HuggingFace computing an internal CE loss we never use
        decoder_input_ids = self.student._shift_right(labels)

        # ── Teacher (clean input, no gradient) ──
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_input_ids=decoder_input_ids,
            )
            t_enc_out = teacher_outputs.encoder_last_hidden_state
            z_T = self.teacher_head(t_enc_out)

            z_T_centered = z_T - self.center.unsqueeze(0).unsqueeze(0)
            p_T = F.softmax(z_T_centered / self._current_tau_t, dim=-1)

            if valid_mask.any():
                batch_center = z_T[valid_mask].mean(dim=0)
                self.center.mul_(cfg.center_momentum).add_(batch_center, alpha=1.0 - cfg.center_momentum)

            t_logits = teacher_outputs.logits

        # ── Student (corrupted input, track gradients) ──
        student_outputs = self.student(
            input_ids=corrupted_ids,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
        )

        s_enc_out = student_outputs.encoder_last_hidden_state
        z_S = self.student_head(s_enc_out)
        p_S_log = F.log_softmax(z_S / cfg.tau_s, dim=-1)

        s_logits = student_outputs.logits

        # ── Loss computation ──
        per_token_dino_loss = -(p_T * p_S_log).sum(dim=-1)
        dino_loss = per_token_dino_loss[valid_mask].mean() if valid_mask.any() else per_token_dino_loss.mean()

        t_probs = F.softmax(t_logits, dim=-1)
        s_log_probs = F.log_softmax(s_logits, dim=-1)
        per_token_kd_loss = F.kl_div(s_log_probs, t_probs, reduction='none').sum(dim=-1)
        target_valid_mask = (labels != -100)
        kd_loss = per_token_kd_loss[target_valid_mask].mean() if target_valid_mask.any() else per_token_kd_loss.mean()

        total_loss = cfg.lambda_dino * dino_loss + cfg.lambda_kd * kd_loss

        # ── Diagnostics ──
        with torch.no_grad():
            ln_K = math.log(cfg.proj_output)
            p_S_probs = F.softmax(z_S / cfg.tau_s, dim=-1)
            H_T = -(p_T * (p_T + 1e-10).log()).sum(dim=-1)
            H_S = -(p_S_probs * (p_S_probs + 1e-10).log()).sum(dim=-1)

            if valid_mask.any():
                H_norm_T = (H_T[valid_mask].mean() / ln_K).item()
                H_norm_S = (H_S[valid_mask].mean() / ln_K).item()
                std_z_T = z_T[valid_mask].std().item()
                std_z_S = z_S[valid_mask].std().item()
            else:
                H_norm_T = (H_T.mean() / ln_K).item()
                H_norm_S = (H_S.mean() / ln_K).item()
                std_z_T = z_T.std().item()
                std_z_S = z_S.std().item()

        return {
            "total_loss": total_loss,
            "dino_loss": dino_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "diagnostics": {
                "H_norm_T": H_norm_T,
                "H_norm_S": H_norm_S,
                "std_z_T": std_z_T,
                "std_z_S": std_z_S,
                "corrupt_ratio": corruption_mask.sum().item() / max(valid_mask.sum().item(), 1),
            },
        }


# ──────────────────────────────────────────────────────────────
# Scheduler helpers
# ──────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_ema_momentum(step: int, total_steps: int, base: float = 0.996) -> float:
    return 1.0 - (1.0 - base) * (math.cos(math.pi * step / total_steps) + 1.0) / 2.0


def get_teacher_temp(step: int, total_steps: int, t_start: float, t_end: float) -> float:
    warmup_steps = int(0.3 * total_steps)
    if step < warmup_steps:
        return t_start + (t_end - t_start) * step / warmup_steps
    return t_end


def gpu_mem_info() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    used = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{used:.1f}/{total:.0f}GB"


def load_translation_samples(
    train_data_path: str, logger: logging.Logger
) -> List[Tuple[str, str]]:
    """train.csv에서 (전처리된_transliteration, translation) 쌍을 로드."""
    if not train_data_path or not os.path.exists(train_data_path):
        logger.warning(f"train_data_path not found: '{train_data_path}' — sample check disabled")
        return []
    df = pd.read_csv(train_data_path, encoding="utf-8")
    preprocessor = OptimizedPreprocessor()
    srcs = preprocessor.preprocess_batch(df["transliteration"].tolist())
    refs = df["translation"].tolist()
    samples = [
        (s, str(r)) for s, r in zip(srcs, refs)
        if s and str(r) not in ("nan", "", "None")
    ]
    logger.info(f"Loaded {len(samples)} labeled samples for translation checks")
    return samples


@torch.no_grad()
def sample_translation_check(
    model: "DINOEMA",
    tokenizer,
    samples: List[Tuple[str, str]],
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    logger: logging.Logger,
    global_step: int,
):
    """랜덤 샘플 하나를 뽑아 student 모델의 현재 번역 출력 vs 정답을 로그."""
    src, ref = random.choice(samples)
    input_text = "translate Akkadian to English: " + src

    enc = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)

    was_training = model.student.training
    model.student.eval()

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        output_ids = model.student.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=128,
            num_beams=2,
            early_stopping=True,
        )

    if was_training:
        model.student.train()

    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    logger.info(
        f"\n{'─'*60}\n"
        f"[Step {global_step}] Translation Sample Check\n"
        f"  SRC : {src[:150]}\n"
        f"  REF : {ref[:150]}\n"
        f"  PRED: {pred[:150]}\n"
        f"{'─'*60}"
    )


# ──────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────

def train(cfg: DINOEMAConfig):
    logger = setup_logging(cfg.output_dir)
    logger.info("=" * 60)
    logger.info("DINO EMA Pretraining + Sequence KD For ByT5 Akkadian (Single GPU)")
    logger.info("=" * 60)
    logger.info(f"  Model  : {cfg.model_path}")
    logger.info(f"  Data   : {cfg.data_path}")
    logger.info(f"  Batch  : {cfg.batch_size} x accum {cfg.grad_accum} = {cfg.batch_size * cfg.grad_accum} effective")
    logger.info(f"  LR     : {cfg.lr}")
    logger.info(f"  bf16   : {cfg.use_bf16}")

    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    dataset = DINOAkkadianDataset(cfg.data_path, tokenizer, cfg.max_length, logger)

    # 레이블 데이터 로드 (번역 샘플 체크용)
    # Kaggle 경로: published_texts.csv와 같은 디렉토리에 train.csv 존재
    # 예: /kaggle/input/deep-past-initiative-machine-translation/train.csv
    #     /kaggle/input/competitions/deep-past-initiative-machine-translation/train.csv
    train_data_path = cfg.train_data_path
    if not train_data_path:
        # data_path(published_texts.csv)의 디렉토리에서 train.csv를 자동 탐색
        candidate = str(Path(cfg.data_path).parent / "train.csv")
        if os.path.exists(candidate):
            train_data_path = candidate
    translation_samples = load_translation_samples(train_data_path, logger)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, cfg.pad_token_id),
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    model = DINOEMA(cfg, logger).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.student.parameters()},
            {"params": model.student_head.parameters()},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )

    batches_per_epoch = len(dataloader)
    total_steps = (batches_per_epoch // cfg.grad_accum) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Mixed precision scaler (bf16 doesn't need GradScaler, but fp16 does)
    use_amp = cfg.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(not cfg.use_bf16 and use_amp))

    logger.info(f"Total optimizer steps: {total_steps} | Warmup: {warmup_steps}")

    # [Fix #5] Cache parameter list for grad clipping — avoids rebuilding list every accum step
    clip_params = list(model.student.parameters()) + list(model.student_head.parameters())

    global_step = 0
    running_dino = 0.0
    running_kd = 0.0
    running_count = 0

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", dynamic_ncols=True)

        for batch_idx, input_ids in enumerate(pbar):
            input_ids = input_ids.to(device)

            # Update teacher temperature
            model._current_tau_t = get_teacher_temp(
                global_step, total_steps, cfg.tau_t_start, cfg.tau_t_end
            )

            # Forward pass (with optional autocast)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                losses = model(input_ids)
                # Scale loss for gradient accumulation
                loss = losses["total_loss"] / cfg.grad_accum

            scaler.scale(loss).backward()

            is_accum_step = (batch_idx + 1) % cfg.grad_accum == 0

            if is_accum_step:
                scaler.unscale_(optimizer)
                # [Fix #5] Use cached param list
                torch.nn.utils.clip_grad_norm_(clip_params, cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                # EMA teacher update
                ema_m = get_ema_momentum(global_step, total_steps, cfg.ema_base)
                model.update_teacher(ema_m)
                # [Fix #1] empty_cache() 제거 — CUDA sync stall 원인

                global_step += 1

            # 번역 샘플 체크 — batch_idx 기준 (tqdm에 보이는 숫자와 일치)
            if translation_samples and (batch_idx + 1) % cfg.sample_check_every == 0:
                sample_translation_check(
                    model, tokenizer, translation_samples,
                    device, amp_dtype, use_amp, logger, global_step,
                )

                if global_step % cfg.log_every_steps == 0:
                    logger.info(
                        f"[Step {global_step}/{total_steps}] "
                        f"DINO={smooth_dino:.4f} KD={smooth_kd:.4f} "
                        f"Total={losses['total_loss'].item():.4f} | "
                        f"EMA={ema_m:.5f} | lr={scheduler.get_last_lr()[0]:.2e} | "
                        f"GPU={gpu_mem_info()}"
                    )

            # Running loss tracking (exponential moving average style)
            running_dino += losses["dino_loss"].item()
            running_kd += losses["kd_loss"].item()
            running_count += 1
            if running_count > 50:
                running_dino -= running_dino / running_count
                running_kd -= running_kd / running_count
                running_count -= 1

            smooth_dino = running_dino / max(1, running_count)
            smooth_kd = running_kd / max(1, running_count)
            ema_m_display = get_ema_momentum(global_step, total_steps, cfg.ema_base)
            pbar.set_postfix_str(
                f"DINO={smooth_dino:.3f} KD={smooth_kd:.3f} EMA={ema_m_display:.4f} GPU={gpu_mem_info()}"
            )

        # ── Epoch checkpoint ──
        epoch_dir = Path(cfg.output_dir) / f"epoch-{epoch + 1}"
        student_dir = epoch_dir / "student"
        teacher_dir = epoch_dir / "teacher"
        student_dir.mkdir(exist_ok=True, parents=True)
        teacher_dir.mkdir(exist_ok=True, parents=True)

        # Student와 Teacher를 분리 저장 — 재학습 시 EMA divergence 보존
        model.student.save_pretrained(student_dir, safe_serialization=False)
        model.teacher.save_pretrained(teacher_dir, safe_serialization=False)
        tokenizer.save_pretrained(epoch_dir)
        torch.save({
            "student_head": model.student_head.state_dict(),
            "teacher_head": model.teacher_head.state_dict(),
            "center": model.center,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
        }, epoch_dir / "dino_state.pt")
        logger.info(f"Epoch {epoch + 1} checkpoint saved → {epoch_dir}")

    # ── Final save ──
    final_dir = Path(cfg.output_dir) / "final"
    student_dir = final_dir / "student"
    teacher_dir = final_dir / "teacher"
    student_dir.mkdir(exist_ok=True, parents=True)
    teacher_dir.mkdir(exist_ok=True, parents=True)
    model.student.save_pretrained(student_dir, safe_serialization=False)
    model.teacher.save_pretrained(teacher_dir, safe_serialization=False)
    tokenizer.save_pretrained(final_dir)
    logger.info("Training complete. Final model saved.")


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
