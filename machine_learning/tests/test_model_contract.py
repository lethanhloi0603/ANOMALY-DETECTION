from __future__ import annotations

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from insider_ml.contracts import PreparedWindows  # noqa: E402
from insider_ml.inference import daily_branch_reconstruction_scores  # noqa: E402
from insider_ml.model import build_model  # noqa: E402
from insider_ml.training import reconstruction_loss  # noqa: E402


def _prepared_windows(
    *,
    feature_dimension: int = 128,
    cyclic_time: bool = True,
) -> PreparedWindows:
    shape = (1, 30, 256)
    return PreparedWindows(
        feature_values=np.zeros((1, 30, feature_dimension), dtype=np.float32),
        feature_mask=np.ones((1, 30, feature_dimension), dtype=bool),
        tokens=np.ones(shape, dtype=np.int64),
        token_mask=np.ones(shape, dtype=bool),
        pc_contexts=np.ones(shape, dtype=np.int64),
        calendar_contexts=np.ones(shape, dtype=np.int64),
        gap_buckets=np.ones(shape, dtype=np.int64),
        time_sin=np.zeros(shape, dtype=np.float32),
        time_cos=(
            np.ones(shape, dtype=np.float32)
            if cyclic_time
            else np.zeros(shape, dtype=np.float32)
        ),
        window_dates=np.asarray(
            [[
                f"2010-05-{day:02d}"
                for day in range(2, 32)
            ]]
        ),
        eligible_day_mask=np.ones((1, 30), dtype=bool),
        sample_user_ids=np.asarray(["U001"]),
        sample_end_days=np.asarray(["2010-05-31"]),
        sample_splits=np.asarray(["TRAIN"]),
        feature_schema_version=np.asarray("feature128.v5"),
        sequence_schema_version=np.asarray("sequence7.v4"),
        preprocessing_checksum=np.asarray("a" * 64),
        scaler_checksum=np.asarray("b" * 64),
        feature_value_space=np.asarray("robust_scaled"),
    )


def test_tcn_transformer_autoencoder_contract() -> None:
    batch_size, days, events = 2, 5, 8
    angles = torch.rand(batch_size, days, events) * (2 * torch.pi)
    batch = {
        "feature_values": torch.randn(batch_size, days, 128),
        "feature_mask": torch.ones(batch_size, days, 128, dtype=torch.bool),
        "tokens": torch.randint(1, 8, (batch_size, days, events)),
        "token_mask": torch.ones(batch_size, days, events, dtype=torch.bool),
        "pc_contexts": torch.randint(1, 5, (batch_size, days, events)),
        "calendar_contexts": torch.randint(1, 3, (batch_size, days, events)),
        "gap_buckets": torch.randint(1, 6, (batch_size, days, events)),
        "time_sin": torch.sin(angles),
        "time_cos": torch.cos(angles),
        "eligible_day_mask": torch.ones(batch_size, days, dtype=torch.bool),
    }
    model = build_model()
    output = model(**batch)
    assert output.feature_reconstruction.shape == (batch_size, days, 128)
    assert output.token_logits.shape == (batch_size, days, events, 8)
    assert output.pc_logits.shape == (batch_size, days, events, 5)
    assert output.calendar_logits.shape == (batch_size, days, events, 3)
    assert output.gap_logits.shape == (batch_size, days, events, 6)
    assert output.time_reconstruction.shape == (batch_size, days, events, 2)
    loss, parts = reconstruction_loss(output, batch)
    assert torch.isfinite(loss)
    assert set(parts) == {"loss", "feature_loss", "sequence_loss"}
    scores = daily_branch_reconstruction_scores(output, batch)
    assert scores.feature.shape == (batch_size, days)
    assert scores.sequence.shape == (batch_size, days)
    assert torch.isfinite(scores.feature).all()
    assert torch.isfinite(scores.sequence).all()


def test_prepared_windows_rejects_bad_feature_dimension() -> None:
    windows = _prepared_windows(feature_dimension=127)
    with pytest.raises(ValueError, match="feature dimension"):
        windows.validate()


def test_prepared_windows_rejects_non_cyclic_event_time() -> None:
    windows = _prepared_windows(cyclic_time=False)
    with pytest.raises(ValueError, match="unit circle"):
        windows.validate()


def test_prepared_windows_retains_eligible_inactive_employee_day() -> None:
    windows = _prepared_windows()
    feature_mask = windows.feature_mask.copy()
    tokens = windows.tokens.copy()
    token_mask = windows.token_mask.copy()
    pc_contexts = windows.pc_contexts.copy()
    calendar_contexts = windows.calendar_contexts.copy()
    gap_buckets = windows.gap_buckets.copy()
    time_cos = windows.time_cos.copy()
    feature_mask[:, 0] = False
    tokens[:, 0] = 0
    token_mask[:, 0] = False
    pc_contexts[:, 0] = 0
    calendar_contexts[:, 0] = 0
    gap_buckets[:, 0] = 0
    time_cos[:, 0] = 0

    replace(
        windows,
        feature_mask=feature_mask,
        tokens=tokens,
        token_mask=token_mask,
        pc_contexts=pc_contexts,
        calendar_contexts=calendar_contexts,
        gap_buckets=gap_buckets,
        time_cos=time_cos,
    ).validate()


def test_prepared_windows_rejects_future_day_and_self_declared_split() -> None:
    windows = _prepared_windows()
    dates = windows.window_dates.copy()
    dates[0, -1] = "2010-06-01"
    with pytest.raises(ValueError, match=r"\[D-29,D\]"):
        replace(windows, window_dates=dates).validate()

    with pytest.raises(ValueError, match="conflicts with locked date split"):
        replace(windows, sample_splits=np.asarray(["TEST"])).validate()
