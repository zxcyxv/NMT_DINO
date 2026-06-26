"""Encoder-only self-supervised learning utilities for ByT5-style models."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token states without letting padding positions affect the result."""
    mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def latent_augment(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    noise_std: float,
    dropout_prob: float,
) -> torch.Tensor:
    """Create a continuous-space text view without replacing discrete byte ids."""
    valid = attention_mask.bool().unsqueeze(-1)
    augmented = embeddings
    if noise_std > 0:
        noise = torch.randn_like(augmented) * noise_std
        augmented = augmented + noise.masked_fill(~valid, 0.0)
    if dropout_prob > 0:
        keep_prob = 1.0 - dropout_prob
        drop_mask = torch.rand_like(augmented).lt(keep_prob)
        augmented = augmented * drop_mask.to(augmented.dtype) / keep_prob
    return augmented


class LargeDINOProjectionHead(nn.Module):
    """DINO-style projection head with a large pseudo-class output space.

    The bottleneck keeps the 65k output setting practical: most parameters live in
    a 256 x K final layer instead of a hidden_dim x K layer.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 65536,
        initial_g: float = 1.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        with torch.no_grad():
            self.last_layer.parametrizations.weight.original0.fill_(initial_g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


class EncoderOnlyDINO(nn.Module):
    """EMA teacher/student DINO wrapper that trains only source-side representations."""

    def __init__(
        self,
        model_path: str,
        d_model: int,
        proj_hidden_dim: int = 2048,
        proj_bottleneck_dim: int = 256,
        proj_out_dim: int = 65536,
        center_momentum: float = 0.9,
        freeze_decoder: bool = True,
        train_embeddings: bool = False,
    ):
        super().__init__()
        self.student = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        if d_model <= 0:
            d_model = getattr(self.student.config, "d_model", None)
            if d_model is None:
                d_model = getattr(self.student.config, "hidden_size", None)
            if d_model is None:
                raise ValueError("d_model could not be inferred from the model config")
        self.teacher = copy.deepcopy(self.student)
        self.student_head = LargeDINOProjectionHead(
            d_model=d_model,
            hidden_dim=proj_hidden_dim,
            bottleneck_dim=proj_bottleneck_dim,
            out_dim=proj_out_dim,
        )
        self.teacher_head = LargeDINOProjectionHead(
            d_model=d_model,
            hidden_dim=proj_hidden_dim,
            bottleneck_dim=proj_bottleneck_dim,
            out_dim=proj_out_dim,
        )
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        if freeze_decoder:
            self._freeze_decoder(self.student)
        if not train_embeddings:
            self._freeze_embeddings(self.student)

        for param in self.teacher.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False

        self.teacher.eval()
        self.teacher_head.eval()
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(proj_out_dim))

    @staticmethod
    def _freeze_decoder(model: nn.Module) -> None:
        decoder = getattr(model, "decoder", None)
        if decoder is not None:
            for param in decoder.parameters():
                param.requires_grad = False
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None:
            for param in lm_head.parameters():
                param.requires_grad = False

    @staticmethod
    def _freeze_embeddings(model: nn.Module) -> None:
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            for param in embeddings.parameters():
                param.requires_grad = False
        shared = getattr(model, "shared", None)
        if shared is not None:
            for param in shared.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_head.eval()
        return self

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for teacher_param, student_param in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tau_student: float,
        tau_teacher: float,
        student_noise_std: float,
        teacher_noise_std: float,
        student_dropout: float,
        teacher_dropout: float,
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        student_embeddings = self.student.get_input_embeddings()(input_ids)
        student_view = latent_augment(
            student_embeddings,
            attention_mask,
            noise_std=student_noise_std,
            dropout_prob=student_dropout,
        )

        student_outputs = self.student.encoder(
            inputs_embeds=student_view,
            attention_mask=attention_mask,
            return_dict=True,
        )
        student_repr = masked_mean_pool(student_outputs.last_hidden_state, attention_mask)
        student_logits = self.student_head(student_repr)
        student_log_probs = F.log_softmax(student_logits / tau_student, dim=-1)

        with torch.no_grad():
            teacher_embeddings = self.teacher.get_input_embeddings()(input_ids)
            teacher_view = latent_augment(
                teacher_embeddings,
                attention_mask,
                noise_std=teacher_noise_std,
                dropout_prob=teacher_dropout,
            )
            teacher_outputs = self.teacher.encoder(
                inputs_embeds=teacher_view,
                attention_mask=attention_mask,
                return_dict=True,
            )
            teacher_repr = masked_mean_pool(teacher_outputs.last_hidden_state, attention_mask)
            teacher_logits = self.teacher_head(teacher_repr)
            teacher_probs = F.softmax((teacher_logits - self.center) / tau_teacher, dim=-1)

            batch_center = teacher_logits.mean(dim=0)
            self.center.mul_(self.center_momentum).add_(
                batch_center,
                alpha=1.0 - self.center_momentum,
            )

        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()

        with torch.no_grad():
            log_k = math.log(student_logits.size(-1))
            student_probs = F.softmax(student_logits / tau_student, dim=-1)
            teacher_entropy = -(teacher_probs * (teacher_probs + 1e-8).log()).sum(dim=-1)
            student_entropy = -(student_probs * (student_probs + 1e-8).log()).sum(dim=-1)
            diagnostics = {
                "teacher_entropy_norm": (teacher_entropy.mean() / log_k).item(),
                "student_entropy_norm": (student_entropy.mean() / log_k).item(),
                "teacher_logit_std": teacher_logits.std().item(),
                "student_logit_std": student_logits.std().item(),
                "center_norm": self.center.norm().item(),
            }

        return {"loss": loss, "diagnostics": diagnostics}
