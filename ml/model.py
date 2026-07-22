from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, dilation=dilation, padding=0)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        y = x.transpose(1, 2)
        y = F.pad(y, (self.left_padding, 0))
        y = self.conv(y)
        y = F.relu(y)
        y = y.transpose(1, 2)
        y = self.dropout(y)
        y = self.norm(y)
        return residual + y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]

class TCNTransformerAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 128, window_size: int = 30, hidden_dim: int = 128,
                 kernel_size: int = 3, dropout: float = 0.15, transformer_layers: int = 2, heads: int = 4):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.tcn = nn.Sequential(*[
            TCNBlock(hidden_dim, hidden_dim, kernel_size, d, dropout) for d in [1, 2, 4, 8]
        ])
        self.positional = PositionalEncoding(hidden_dim, max_len=window_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, activation="relu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_projection(x)
        z = self.tcn(z)
        z = self.positional(z)
        z = self.transformer(z)
        return self.decoder(z)


class DaySequenceEncoder(nn.Module):
    """Encode raw daily token/source/time-gap sequences into R^64.

    This is intentionally different from feeding engineered token counts into
    PCA: token, source, time gap and position are learned jointly. A local
    convolution preserves order-sensitive patterns before attention pooling.
    """

    def __init__(
        self,
        vocab_size: int,
        source_vocab_size: int,
        max_events_per_day: int,
        output_dim: int = 64,
        token_embedding_dim: int = 32,
        source_embedding_dim: int = 8,
        gap_embedding_dim: int = 8,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.max_events_per_day = max_events_per_day
        self.token_embedding = nn.Embedding(vocab_size, token_embedding_dim, padding_idx=0)
        self.source_embedding = nn.Embedding(source_vocab_size, source_embedding_dim)
        self.gap_projection = nn.Sequential(
            nn.Linear(1, gap_embedding_dim),
            nn.ReLU(),
        )
        event_dim = token_embedding_dim + source_embedding_dim + gap_embedding_dim
        self.event_projection = nn.Linear(event_dim, output_dim)
        self.position_embedding = nn.Embedding(max_events_per_day, output_dim)
        self.local_encoder = nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)
        self.attention = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        source_ids: torch.Tensor,
        time_gaps: torch.Tensor,
    ) -> torch.Tensor:
        # Inputs: [batch, calendar_days, events]
        batch, days, events = token_ids.shape
        flat_tokens = token_ids.reshape(batch * days, events)
        flat_sources = source_ids.reshape(batch * days, events)
        flat_gaps = time_gaps.reshape(batch * days, events)
        active = flat_tokens.ne(0)

        token_embedding = self.token_embedding(flat_tokens)
        source_embedding = self.source_embedding(flat_sources)
        gap_embedding = self.gap_projection(torch.log1p(flat_gaps.clamp_min(0)).unsqueeze(-1))
        event_embedding = torch.cat([token_embedding, source_embedding, gap_embedding], dim=-1)
        event_embedding = self.event_projection(event_embedding)
        positions = torch.arange(events, device=token_ids.device).unsqueeze(0)
        event_embedding = event_embedding + self.position_embedding(positions)
        event_embedding = event_embedding * active.unsqueeze(-1)

        local = self.local_encoder(event_embedding.transpose(1, 2)).transpose(1, 2)
        local = self.norm(F.relu(local) + event_embedding)
        local = self.dropout(local) * active.unsqueeze(-1)

        logits = self.attention(local).squeeze(-1).masked_fill(~active, -1e9)
        weights = torch.softmax(logits, dim=-1) * active
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        pooled = torch.sum(local * weights.unsqueeze(-1), dim=1)
        return pooled.reshape(batch, days, -1)


class MultiViewTCNTransformerAutoencoder(nn.Module):
    """Count projection + raw daily sequence encoder + temporal AE."""

    def __init__(
        self,
        count_dim: int = 64,
        sequence_dim: int = 64,
        window_size: int = 30,
        vocab_size: int = 25,
        source_vocab_size: int = 6,
        max_events_per_day: int = 256,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.15,
        transformer_layers: int = 2,
        heads: int = 4,
        view_mode: str = "count-sequence",
    ):
        super().__init__()
        if view_mode not in {"count", "sequence", "count-sequence"}:
            raise ValueError(f"Unsupported view_mode: {view_mode}")
        self.count_dim = count_dim
        self.sequence_dim = sequence_dim
        self.view_mode = view_mode
        self.sequence_encoder = DaySequenceEncoder(
            vocab_size=vocab_size,
            source_vocab_size=source_vocab_size,
            max_events_per_day=max_events_per_day,
            output_dim=sequence_dim,
            dropout=dropout,
        )
        self.temporal_autoencoder = TCNTransformerAutoencoder(
            input_dim=count_dim + sequence_dim,
            window_size=window_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            transformer_layers=transformer_layers,
            heads=heads,
        )

    def forward(
        self,
        count_window: torch.Tensor,
        token_window: torch.Tensor,
        source_window: torch.Tensor,
        gap_window: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_day = self.sequence_encoder(token_window, source_window, gap_window)
        count_day = count_window
        if self.view_mode == "count":
            sequence_day = torch.zeros_like(sequence_day)
        elif self.view_mode == "sequence":
            count_day = torch.zeros_like(count_day)
        day_representation = torch.cat([count_day, sequence_day], dim=-1)
        reconstruction = self.temporal_autoencoder(day_representation)
        return reconstruction, day_representation
