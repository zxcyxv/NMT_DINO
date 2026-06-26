#!/usr/bin/env python3
"""Encoder-only SSL pretraining for Old Assyrian ByT5 models.

This is the cleaner follow-up to the original token-wise DINO experiments:

- no random byte replacement
- no decoder KL/reconstruction objective during SSL
- two continuous latent-space views of the same source text
- sentence-level pooled encoder representation
- large DINO pseudo-class projection space

The resulting checkpoint is intended for a second supervised NMT fine-tuning
stage on Akkadian-English parallel data.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nmt_dino.encoder_ssl import EncoderOnlyDINO
from nmt_dino.preprocessing import OptimizedPreprocessor


@dataclass
class EncoderSSLConfig:
    model_path: str = "google/byt5-small"
    data_path: str = "dataset/published_texts.csv"
    output_dir: str = "outputs/encoder_ssl"
    text_column: str = "transliteration"
    source_prefix: str = ""

    d_model: int = 0
    proj_hidden_dim: int = 2048
    proj_bottleneck_dim: int = 256
    proj_out_dim: int = 65536

    batch_size: int = 8
    grad_accum: int = 4
    epochs: int = 5
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    max_length: int = 512
    num_workers: int = 2

    tau_student: float = 0.1
    tau_teacher_start: float = 0.04
    tau_teacher_end: float = 0.07
    ema_base: float = 0.996
    center_momentum: float = 0.9

    student_noise_std: float = 0.02
    teacher_noise_std: float = 0.01
    student_dropout: float = 0.10
    teacher_dropout: float = 0.05

    freeze_decoder: bool = True
    train_embeddings: bool = False
    gradient_checkpointing: bool = True
    use_bf16: bool = True
    seed: int = 42
    log_every_steps: int = 10
    save_every_epochs: int = 1

    device: str = "cuda"


def parse_args() -> EncoderSSLConfig:
    defaults = EncoderSSLConfig()
    parser = argparse.ArgumentParser(description="Encoder-only latent-view DINO SSL for ByT5")
    for key, value in asdict(defaults).items():
        value_type = type(value)
        if value_type is bool:
            parser.add_argument(
                f"--{key}",
                type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
                default=value,
            )
        else:
            parser.add_argument(f"--{key}", type=value_type, default=value)
    return EncoderSSLConfig(**vars(parser.parse_args()))


def setup_logging(output_dir: str) -> logging.Logger:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("encoder_ssl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(Path(output_dir) / "encoder_ssl.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TransliterationDataset(Dataset):
    def __init__(self, cfg: EncoderSSLConfig, tokenizer, logger: logging.Logger):
        df = pd.read_csv(cfg.data_path, encoding="utf-8")
        if cfg.text_column not in df.columns:
            raise ValueError(f"Column '{cfg.text_column}' not found in {cfg.data_path}")

        preprocessor = OptimizedPreprocessor()
        texts = preprocessor.preprocess_batch(df[cfg.text_column].tolist())
        texts = [cfg.source_prefix + text for text in texts if text and text.strip()]
        logger.info("Loaded %d usable unlabeled texts from %s", len(texts), cfg.data_path)

        self.input_ids: list[torch.Tensor] = []
        for text in tqdm(texts, desc="Tokenizing", dynamic_ncols=True):
            encoded = tokenizer(
                text,
                max_length=cfg.max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            self.input_ids.append(encoded.input_ids.squeeze(0))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.input_ids[index]


def collate_input_ids(batch: list[torch.Tensor], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(item.size(0) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for row, item in enumerate(batch):
        input_ids[row, : item.size(0)] = item
    attention_mask = input_ids.ne(pad_token_id).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def teacher_temperature(step: int, total_steps: int, start: float, end: float) -> float:
    warmup_steps = max(1, int(total_steps * 0.3))
    if step >= warmup_steps:
        return end
    return start + (end - start) * step / warmup_steps


def ema_momentum(step: int, total_steps: int, base: float) -> float:
    if total_steps <= 1:
        return 1.0
    return 1.0 - (1.0 - base) * (math.cos(math.pi * step / total_steps) + 1.0) / 2.0


def save_checkpoint(
    cfg: EncoderSSLConfig,
    model: EncoderOnlyDINO,
    tokenizer,
    output_dir: Path,
    epoch: int,
    global_step: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    student_dir = output_dir / "student"
    teacher_dir = output_dir / "teacher"
    student_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)

    model.student.save_pretrained(student_dir, safe_serialization=False)
    model.teacher.save_pretrained(teacher_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)
    torch.save(
        {
            "student_head": model.student_head.state_dict(),
            "teacher_head": model.teacher_head.state_dict(),
            "center": model.center.detach().cpu(),
            "epoch": epoch,
            "global_step": global_step,
            "config": asdict(cfg),
        },
        output_dir / "encoder_ssl_state.pt",
    )
    with open(output_dir / "encoder_ssl_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)


def train(cfg: EncoderSSLConfig) -> None:
    logger = setup_logging(cfg.output_dir)
    set_seed(cfg.seed)

    device = torch.device(cfg.device if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_bf16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if cfg.use_bf16 else torch.float16

    logger.info("Encoder-only latent-view DINO SSL")
    logger.info("Model: %s", cfg.model_path)
    logger.info("Data : %s", cfg.data_path)
    logger.info("Device: %s | bf16=%s", device, use_amp)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    dataset = TransliterationDataset(cfg, tokenizer, logger)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda batch: collate_input_ids(batch, pad_token_id),
    )
    if len(dataloader) == 0:
        raise ValueError("No training batches. Reduce batch_size or check the input data.")

    model = EncoderOnlyDINO(
        model_path=cfg.model_path,
        d_model=cfg.d_model,
        proj_hidden_dim=cfg.proj_hidden_dim,
        proj_bottleneck_dim=cfg.proj_bottleneck_dim,
        proj_out_dim=cfg.proj_out_dim,
        center_momentum=cfg.center_momentum,
        freeze_decoder=cfg.freeze_decoder,
        train_embeddings=cfg.train_embeddings,
    ).to(device)

    if cfg.gradient_checkpointing and hasattr(model.student, "gradient_checkpointing_enable"):
        model.student.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for the student model")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info("Trainable parameters: %d", sum(p.numel() for p in trainable_params))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = max(1, (len(dataloader) // cfg.grad_accum) * cfg.epochs)
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not cfg.use_bf16))

    logger.info("Optimizer steps: %d | warmup: %d", total_steps, warmup_steps)
    global_step = 0
    running_loss = 0.0
    running_count = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.epochs}", dynamic_ncols=True)

        for batch_index, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            tau_teacher = teacher_temperature(
                global_step,
                total_steps,
                cfg.tau_teacher_start,
                cfg.tau_teacher_end,
            )

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                result = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    tau_student=cfg.tau_student,
                    tau_teacher=tau_teacher,
                    student_noise_std=cfg.student_noise_std,
                    teacher_noise_std=cfg.teacher_noise_std,
                    student_dropout=cfg.student_dropout,
                    teacher_dropout=cfg.teacher_dropout,
                )
                loss = result["loss"] / cfg.grad_accum

            scaler.scale(loss).backward()
            running_loss += result["loss"].item()
            running_count += 1

            if batch_index % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                momentum = ema_momentum(global_step, total_steps, cfg.ema_base)
                model.update_teacher(momentum)
                global_step += 1

                if global_step % cfg.log_every_steps == 0:
                    diagnostics = result["diagnostics"]
                    avg_loss = running_loss / max(1, running_count)
                    logger.info(
                        "step=%d/%d loss=%.4f tau_t=%.4f ema=%.5f "
                        "H_t=%.3f H_s=%.3f std_t=%.3f std_s=%.3f center=%.2f lr=%.2e",
                        global_step,
                        total_steps,
                        avg_loss,
                        tau_teacher,
                        momentum,
                        diagnostics["teacher_entropy_norm"],
                        diagnostics["student_entropy_norm"],
                        diagnostics["teacher_logit_std"],
                        diagnostics["student_logit_std"],
                        diagnostics["center_norm"],
                        scheduler.get_last_lr()[0],
                    )
                    running_loss = 0.0
                    running_count = 0

            progress.set_postfix(loss=f"{result['loss'].item():.3f}", step=global_step)

        if cfg.save_every_epochs > 0 and epoch % cfg.save_every_epochs == 0:
            save_checkpoint(
                cfg,
                model,
                tokenizer,
                Path(cfg.output_dir) / f"epoch-{epoch}",
                epoch=epoch,
                global_step=global_step,
            )
            logger.info("Saved epoch checkpoint: %s", Path(cfg.output_dir) / f"epoch-{epoch}")

    save_checkpoint(
        cfg,
        model,
        tokenizer,
        Path(cfg.output_dir) / "final",
        epoch=cfg.epochs,
        global_step=global_step,
    )
    logger.info("Training complete. Final checkpoint: %s", Path(cfg.output_dir) / "final")


if __name__ == "__main__":
    train(parse_args())
