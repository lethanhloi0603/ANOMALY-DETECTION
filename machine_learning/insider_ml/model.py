"""Compact TCN-Transformer Autoencoder for feature and behavior branches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from insider_ml.contracts import (
    FEATURE_DIMENSION,
    FEATURE_SCHEMA_VERSION,
    GAP_BUCKETS,
    MAX_SEQUENCE_LENGTH,
    SEQUENCE_TOKENS,
)


@dataclass(slots=True)
class AutoencoderOutput:
    feature_reconstruction: torch.Tensor
    token_logits: torch.Tensor
    pc_logits: torch.Tensor
    calendar_logits: torch.Tensor
    gap_logits: torch.Tensor
    time_reconstruction: torch.Tensor
    latent_days: torch.Tensor
    day_mask: torch.Tensor


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.network(values)
        return self.normalization((values + residual).transpose(1, 2)).transpose(1, 2)


class SequenceEncoder(nn.Module):
    """Order-aware encoder over the events inside each user-day."""

    def __init__(self, output_dimension: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(len(SEQUENCE_TOKENS), 24, padding_idx=0)
        self.pc_embedding = nn.Embedding(5, 4, padding_idx=0)
        self.calendar_embedding = nn.Embedding(3, 3, padding_idx=0)
        self.gap_embedding = nn.Embedding(len(GAP_BUCKETS), 4, padding_idx=0)
        self.event_tcn = nn.Sequential(
            nn.Conv1d(37, output_dimension, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(output_dimension, output_dimension, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        pc_contexts: torch.Tensor,
        calendar_contexts: torch.Tensor,
        gap_buckets: torch.Tensor,
        time_sin: torch.Tensor,
        time_cos: torch.Tensor,
    ) -> torch.Tensor:
        batch, days, events = tokens.shape
        embedded = torch.cat(
            (
                self.token_embedding(tokens),
                self.pc_embedding(pc_contexts),
                self.calendar_embedding(calendar_contexts),
                self.gap_embedding(gap_buckets),
                time_sin.unsqueeze(-1).to(torch.float32),
                time_cos.unsqueeze(-1).to(torch.float32),
            ),
            dim=-1,
        )
        flattened = embedded.reshape(batch * days, events, -1).transpose(1, 2)
        encoded = self.event_tcn(flattened).transpose(1, 2)
        mask = token_mask.reshape(batch * days, events, 1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled.reshape(batch, days, -1)


class TCNTransformerAutoencoder(nn.Module):
    def __init__(
        self,
        *,
        window_days: int = 30,
        feature_dimension: int = FEATURE_DIMENSION,
        feature_embedding_dimension: int = 64,
        sequence_embedding_dimension: int = 64,
        day_embedding_dimension: int = 128,
        sequence_max_len: int = MAX_SEQUENCE_LENGTH,
        tcn_channels: int = 128,
        tcn_kernel_size: int = 3,
        tcn_dilations: tuple[int, ...] = (1, 2, 4, 8),
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_feed_forward_dimension: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dimension != FEATURE_DIMENSION:
            raise ValueError(
                f"model requires {FEATURE_SCHEMA_VERSION} ({FEATURE_DIMENSION} dimensions)"
            )
        if sequence_max_len > MAX_SEQUENCE_LENGTH:
            raise ValueError(f"sequence_max_len cannot exceed {MAX_SEQUENCE_LENGTH}")
        if tcn_channels % transformer_heads:
            raise ValueError("tcn_channels must be divisible by transformer_heads")

        self.feature_dimension = feature_dimension
        self.sequence_max_len = sequence_max_len
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dimension * 2, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, feature_embedding_dimension),
        )
        self.sequence_encoder = SequenceEncoder(sequence_embedding_dimension)
        self.day_fusion = nn.Linear(
            feature_embedding_dimension + sequence_embedding_dimension,
            day_embedding_dimension,
        )
        self.tcn_input = nn.Linear(day_embedding_dimension, tcn_channels)
        self.day_positions = nn.Embedding(window_days, tcn_channels)
        self.tcn = nn.ModuleList(
            ResidualTCNBlock(tcn_channels, tcn_kernel_size, dilation, dropout)
            for dilation in tcn_dilations
        )
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=tcn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_feed_forward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, transformer_layers)
        self.feature_decoder = nn.Sequential(
            nn.Linear(tcn_channels, 128),
            nn.GELU(),
            nn.Linear(128, feature_dimension),
        )
        self.event_positions = nn.Embedding(sequence_max_len, tcn_channels)
        self.token_decoder = nn.Sequential(
            nn.Linear(tcn_channels, 64),
            nn.GELU(),
            nn.Linear(64, len(SEQUENCE_TOKENS)),
        )
        self.pc_decoder = nn.Linear(tcn_channels, 5)
        self.calendar_decoder = nn.Linear(tcn_channels, 3)
        self.gap_decoder = nn.Linear(tcn_channels, len(GAP_BUCKETS))
        self.time_decoder = nn.Sequential(nn.Linear(tcn_channels, 2), nn.Tanh())

    def forward(
        self,
        *,
        feature_values: torch.Tensor,
        feature_mask: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        pc_contexts: torch.Tensor,
        calendar_contexts: torch.Tensor,
        gap_buckets: torch.Tensor,
        time_sin: torch.Tensor,
        time_cos: torch.Tensor,
        eligible_day_mask: torch.Tensor,
    ) -> AutoencoderOutput:
        batch, days, feature_count = feature_values.shape
        if feature_count != self.feature_dimension:
            raise ValueError(f"expected {self.feature_dimension} feature values")
        if tokens.shape[:2] != (batch, days) or tokens.shape[-1] > self.sequence_max_len:
            raise ValueError("sequence tensor is not aligned with the feature window")
        if eligible_day_mask.shape != (batch, days):
            raise ValueError("eligible_day_mask must align with the feature window")

        observed_values = feature_values * feature_mask.to(feature_values.dtype)
        feature_input = torch.cat((observed_values, feature_mask.to(feature_values.dtype)), dim=-1)
        feature_embedding = self.feature_encoder(feature_input)
        sequence_embedding = self.sequence_encoder(
            tokens,
            token_mask,
            pc_contexts,
            calendar_contexts,
            gap_buckets,
            time_sin,
            time_cos,
        )
        latent = self.tcn_input(
            self.day_fusion(torch.cat((feature_embedding, sequence_embedding), dim=-1))
        )
        positions = torch.arange(days, device=latent.device)
        latent = latent + self.day_positions(positions)[None, :, :]

        temporal = latent.transpose(1, 2)
        for block in self.tcn:
            temporal = block(temporal)
        temporal = temporal.transpose(1, 2)

        day_mask = eligible_day_mask.to(torch.bool)
        safe_day_mask = day_mask.clone()
        safe_day_mask[~safe_day_mask.any(dim=1), 0] = True
        latent_days = self.transformer(temporal, src_key_padding_mask=~safe_day_mask)
        feature_reconstruction = self.feature_decoder(latent_days)

        event_count = tokens.shape[-1]
        event_positions = self.event_positions(
            torch.arange(event_count, device=latent_days.device)
        )
        token_latent = latent_days[:, :, None, :] + event_positions[None, None, :, :]
        token_logits = self.token_decoder(token_latent)
        return AutoencoderOutput(
            feature_reconstruction=feature_reconstruction,
            token_logits=token_logits,
            pc_logits=self.pc_decoder(token_latent),
            calendar_logits=self.calendar_decoder(token_latent),
            gap_logits=self.gap_decoder(token_latent),
            time_reconstruction=self.time_decoder(token_latent),
            latent_days=latent_days,
            day_mask=day_mask,
        )


def load_model_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[1] / "config" / "model.tcn_transformer_ae.v4.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "tcn-transformer-ae.v4":
        raise ValueError(f"unsupported model config: {config_path}")
    if payload.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("model config feature schema is incompatible")
    if payload.get("sequence_schema_version") != "sequence7.v4":
        raise ValueError("model config sequence schema is incompatible")
    return payload


def build_model(config: dict[str, Any] | None = None) -> TCNTransformerAutoencoder:
    values = config or load_model_config()
    tcn = values["tcn"]
    transformer = values["transformer"]
    return TCNTransformerAutoencoder(
        window_days=int(values["window_days"]),
        feature_dimension=int(values["feature_dimension"]),
        feature_embedding_dimension=int(values["feature_embedding_dimension"]),
        sequence_embedding_dimension=int(values["sequence_embedding_dimension"]),
        day_embedding_dimension=int(values["day_embedding_dimension"]),
        sequence_max_len=int(values["sequence_max_len"]),
        tcn_channels=int(tcn["channels"]),
        tcn_kernel_size=int(tcn["kernel_size"]),
        tcn_dilations=tuple(int(value) for value in tcn["dilations"]),
        transformer_layers=int(transformer["layers"]),
        transformer_heads=int(transformer["heads"]),
        transformer_feed_forward_dimension=int(transformer["feed_forward_dimension"]),
        dropout=float(transformer["dropout"]),
    )
