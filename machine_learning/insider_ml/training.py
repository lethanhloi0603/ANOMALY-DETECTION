"""Training primitives kept independent from orchestration and storage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch.nn import functional as functional

from insider_ml.model import AutoencoderOutput, TCNTransformerAutoencoder


def reconstruction_loss(
    output: AutoencoderOutput,
    batch: Mapping[str, torch.Tensor],
    *,
    feature_weight: float = 0.5,
    sequence_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    feature_mask = batch["feature_mask"].to(output.feature_reconstruction.dtype)
    squared_error = (output.feature_reconstruction - batch["feature_values"]).square()
    feature_loss = (squared_error * feature_mask).sum() / feature_mask.sum().clamp_min(1.0)

    token_mask = batch["token_mask"] & batch["tokens"].ne(0)
    sequence_components: list[torch.Tensor] = []
    if token_mask.any():
        sequence_components.append(
            functional.cross_entropy(
                output.token_logits[token_mask],
                batch["tokens"][token_mask],
            )
        )
        for logits, target_name in (
            (output.pc_logits, "pc_contexts"),
            (output.calendar_logits, "calendar_contexts"),
            (output.gap_logits, "gap_buckets"),
        ):
            target = batch[target_name]
            side_mask = token_mask & target.ne(0)
            if side_mask.any():
                sequence_components.append(
                    functional.cross_entropy(logits[side_mask], target[side_mask])
                )
        target_time = torch.stack((batch["time_sin"], batch["time_cos"]), dim=-1)
        sequence_components.append(
            (output.time_reconstruction[token_mask] - target_time[token_mask]).square().mean()
        )
    if sequence_components:
        sequence_loss = torch.stack(sequence_components).mean()
    else:
        sequence_loss = output.token_logits.sum() * 0.0
    total = feature_weight * feature_loss + sequence_weight * sequence_loss
    return total, {
        "loss": float(total.detach()),
        "feature_loss": float(feature_loss.detach()),
        "sequence_loss": float(sequence_loss.detach()),
    }


def train_epoch(
    model: TCNTransformerAutoencoder,
    batches: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    feature_weight: float = 0.5,
    sequence_weight: float = 0.5,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "feature_loss": 0.0, "sequence_loss": 0.0}
    count = 0
    for raw_batch in batches:
        batch = {name: value.to(device) for name, value in raw_batch.items()}
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss, metrics = reconstruction_loss(
            output,
            batch,
            feature_weight=feature_weight,
            sequence_weight=sequence_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for name, value in metrics.items():
            totals[name] += value
        count += 1
    if count == 0:
        raise ValueError("training loader produced no batches")
    return {name: value / count for name, value in totals.items()}
