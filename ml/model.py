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
