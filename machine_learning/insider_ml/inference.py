"""Reconstruction anomaly scores and immutable hierarchical references."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as functional

from insider_ml.model import AutoencoderOutput


@dataclass(frozen=True, slots=True)
class ReferenceStats:
    level: str
    scope_key: str
    location: float
    scale: float
    observation_count: int
    sorted_scores: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BranchReconstructionScores:
    """Uncalibrated branch errors; each branch gets its own reference/CDF."""

    feature: torch.Tensor
    sequence: torch.Tensor


def daily_branch_reconstruction_scores(
    output: AutoencoderOutput,
    batch: dict[str, torch.Tensor],
) -> BranchReconstructionScores:
    feature_mask = batch["feature_mask"].to(output.feature_reconstruction.dtype)
    feature_error = (
        (output.feature_reconstruction - batch["feature_values"]).square() * feature_mask
    ).sum(dim=-1) / feature_mask.sum(dim=-1).clamp_min(1.0)

    token_loss = functional.cross_entropy(
        output.token_logits.movedim(-1, 1),
        batch["tokens"],
        reduction="none",
    )
    token_mask_bool = batch["token_mask"] & batch["tokens"].ne(0)
    token_mask = token_mask_bool.to(token_loss.dtype)
    sequence_components = [
        (token_loss * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1.0)
    ]
    for logits, target_name in (
        (output.pc_logits, "pc_contexts"),
        (output.calendar_logits, "calendar_contexts"),
        (output.gap_logits, "gap_buckets"),
    ):
        targets = batch[target_name]
        side_loss = functional.cross_entropy(
            logits.movedim(-1, 1),
            targets,
            reduction="none",
        )
        side_mask_bool = token_mask_bool & targets.ne(0)
        side_mask = side_mask_bool.to(side_loss.dtype)
        sequence_components.append(
            (side_loss * side_mask).sum(dim=-1) / side_mask.sum(dim=-1).clamp_min(1.0)
        )
    target_time = torch.stack((batch["time_sin"], batch["time_cos"]), dim=-1)
    time_loss = (output.time_reconstruction - target_time).square().mean(dim=-1)
    sequence_components.append(
        (time_loss * token_mask).sum(dim=-1) / token_mask.sum(
            dim=-1
        ).clamp_min(1.0)
    )
    sequence_error = torch.stack(sequence_components).mean(dim=0)
    eligible = output.day_mask
    feature_available = batch["feature_mask"].any(dim=-1)
    sequence_available = token_mask_bool.any(dim=-1)
    return BranchReconstructionScores(
        feature=feature_error.masked_fill(~(eligible & feature_available), torch.nan),
        sequence=sequence_error.masked_fill(~(eligible & sequence_available), torch.nan),
    )


def fit_reference(level: str, scope_key: str, scores: np.ndarray) -> ReferenceStats:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot fit a reference without finite training scores")
    location = float(np.median(finite))
    mad = float(np.median(np.abs(finite - location)))
    return ReferenceStats(
        level=level.upper(),
        scope_key=scope_key,
        location=location,
        scale=max(1.4826 * mad, 1e-6),
        observation_count=int(finite.size),
        sorted_scores=tuple(float(value) for value in np.sort(finite)),
    )


def calibrated_tail_score(raw_score: float, reference: ReferenceStats) -> float:
    """Return the leakage-safe empirical CDF percentile for a raw score."""

    if not np.isfinite(raw_score):
        raise ValueError("raw_score must be finite")
    if not reference.sorted_scores:
        raise ValueError("reference requires sorted training scores")
    return bisect_right(reference.sorted_scores, float(raw_score)) / len(
        reference.sorted_scores
    )
