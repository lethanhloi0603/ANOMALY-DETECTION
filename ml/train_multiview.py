from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset, Subset

from model import MultiViewTCNTransformerAutoencoder
from multiview_features import (
    COUNT_FEATURE_COLUMNS,
    SOURCE_TO_ID,
    TOKEN_VOCAB,
    assign_temporal_splits,
    calendarize_multiview,
    configure_feature_rules,
    load_ldap_context,
    make_multiview_day_inputs,
    multiview_from_events,
    read_events_from_sqlite,
)


ARCHITECTURE = "multiview-token-encoder-v1"
COUNT_DIM = 64
SEQUENCE_DIM = 64


def safe_signed_log_matrix(multiview: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Convert count/deviation columns to a finite signed-log matrix."""
    if not columns:
        return np.zeros((len(multiview), 0), dtype=np.float64)
    work = multiview[columns].copy()
    for column in columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    values = work.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float64)
    values = np.clip(values, -1e12, 1e12)
    values = np.sign(values) * np.log1p(np.abs(values))
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["global", "personalized"], required=True)
    parser.add_argument("--multiview-csv")
    parser.add_argument("--db")
    parser.add_argument("--user-id")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--global-model-dir")
    parser.add_argument("--role-context")
    parser.add_argument("--feature-rules")
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-events-per-day", type=int, default=256)
    parser.add_argument("--view-mode", choices=["count", "sequence", "count-sequence"], default="count-sequence")
    parser.add_argument("--min-safe-days", "--min-active-days", dest="min_safe_days", type=int, default=30)
    parser.add_argument("--anomaly-quantile", type=float, default=0.995)
    parser.add_argument("--min-role-users", type=int, default=30)
    parser.add_argument("--min-role-user-days", type=int, default=1000)
    parser.add_argument("--safe-history-days", type=int, default=90)
    parser.add_argument("--update-delay-days", type=int, default=7)
    parser.add_argument("--update-every-safe-days", type=int, default=7)
    parser.add_argument("--update-every-calendar-days", type=int, default=7)
    parser.add_argument("--force-update", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--training-label-policy",
        choices=["ignore", "exclude-positive-windows"],
        default="ignore",
        help="Keep labels evaluation-only, or explicitly build a clean one-class training ablation.",
    )
    parser.add_argument(
        "--calibration-label-policy",
        choices=["ignore", "benign-only"],
        default="ignore",
        help="Whether validation positives may influence the fitted anomaly threshold.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def load_multiview(args) -> tuple[pd.DataFrame, int]:
    if args.scope == "global":
        if not args.multiview_csv:
            raise ValueError("--multiview-csv is required")
        path = Path(args.multiview_csv)
        if not path.exists():
            raise FileNotFoundError(f"Multiview file not found: {path}")
        multiview = pd.read_csv(path, low_memory=False)
    else:
        if not args.db or not args.user_id:
            raise ValueError("--db and --user-id are required for personalized calibration")
        role_context = load_ldap_context(args.role_context)
        events = read_events_from_sqlite(args.db, user_id=args.user_id, role_context=role_context)
        multiview = multiview_from_events(events)
    active_days = int(multiview["date"].nunique()) if not multiview.empty else 0
    return multiview, active_days


def pad64(values: np.ndarray) -> np.ndarray:
    if values.shape[1] < COUNT_DIM:
        values = np.hstack([values, np.zeros((values.shape[0], COUNT_DIM - values.shape[1]), dtype=values.dtype)])
    return values[:, :COUNT_DIM]


def fit_or_transform_count_view(multiview: pd.DataFrame, args) -> tuple[np.ndarray, dict]:
    for column in COUNT_FEATURE_COLUMNS:
        if column not in multiview.columns:
            multiview[column] = 0
    raw = safe_signed_log_matrix(multiview, COUNT_FEATURE_COLUMNS)
    if args.scope == "global":
        train_mask = multiview["split"].eq("train").to_numpy()
        if getattr(args, "training_label_policy", "ignore") == "exclude-positive-windows":
            label_series = multiview["label_day"] if "label_day" in multiview else pd.Series(0, index=multiview.index)
            labels = pd.to_numeric(label_series, errors="coerce").fillna(0).to_numpy()
            train_mask &= labels == 0
        if not train_mask.any():
            raise ValueError("Temporal split produced no training rows")
        scaler = RobustScaler()
        scaler.fit(raw[train_mask])
        scaled_train = scaler.transform(raw[train_mask])
        component_count = min(COUNT_DIM, scaled_train.shape[0], scaled_train.shape[1])
        pca = PCA(n_components=component_count, random_state=42)
        pca.fit(scaled_train)
        transformed = pad64(pca.transform(scaler.transform(raw))).astype(np.float32)
        vectorizer = {
            "architecture": ARCHITECTURE,
            "count_scaler": scaler,
            "count_pca": pca,
            "count_cols": COUNT_FEATURE_COLUMNS,
            "count_dim": COUNT_DIM,
        }
        return transformed, vectorizer
    global_dir = Path(args.global_model_dir or "../models/global")
    vectorizer = joblib.load(global_dir / "vectorizer.joblib")
    columns = vectorizer["count_cols"]
    for column in columns:
        if column not in multiview.columns:
            multiview[column] = 0
    raw = safe_signed_log_matrix(multiview, columns)
    transformed = vectorizer["count_pca"].transform(vectorizer["count_scaler"].transform(raw))
    return pad64(transformed).astype(np.float32), vectorizer


def model_config(args) -> dict:
    return {
        "count_dim": COUNT_DIM,
        "sequence_dim": SEQUENCE_DIM,
        "window_size": args.window_size,
        "vocab_size": len(TOKEN_VOCAB),
        "source_vocab_size": len(SOURCE_TO_ID),
        "max_events_per_day": args.max_events_per_day,
        "hidden_dim": args.hidden_dim,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "transformer_layers": args.transformer_layers,
        "heads": args.heads,
        "view_mode": args.view_mode,
    }


class LazyCalendarWindowDataset(Dataset):
    """Materialize one sliding window on demand instead of duplicating all days."""

    def __init__(
        self,
        count_days: np.ndarray,
        token_days: np.ndarray,
        source_days: np.ndarray,
        gap_days: np.ndarray,
        window_slices: list[tuple[int, int]],
        window_size: int,
    ):
        self.count_days = count_days
        self.token_days = token_days
        self.source_days = source_days
        self.gap_days = gap_days
        self.window_slices = window_slices
        self.window_size = int(window_size)

    def __len__(self) -> int:
        return len(self.window_slices)

    @property
    def storage_bytes(self) -> int:
        return int(
            self.count_days.nbytes
            + self.token_days.nbytes
            + self.source_days.nbytes
            + self.gap_days.nbytes
        )

    def __getitem__(self, index: int):
        start, end = self.window_slices[index]
        observed_days = end - start
        destination = self.window_size - observed_days
        event_count = self.token_days.shape[1]
        count_dim = self.count_days.shape[1]

        count = np.zeros((self.window_size, count_dim), dtype=np.float32)
        tokens = np.zeros((self.window_size, event_count), dtype=np.int64)
        sources = np.full(
            (self.window_size, event_count),
            SOURCE_TO_ID["unknown"],
            dtype=np.int64,
        )
        gaps = np.zeros((self.window_size, event_count), dtype=np.float32)
        count[destination:] = self.count_days[start:end]
        tokens[destination:] = self.token_days[start:end]
        sources[destination:] = self.source_days[start:end]
        gaps[destination:] = self.gap_days[start:end]
        return (
            torch.from_numpy(count),
            torch.from_numpy(tokens),
            torch.from_numpy(sources),
            torch.from_numpy(gaps),
        )


def reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor, view_mode: str) -> torch.Tensor:
    count_loss = torch.nn.functional.mse_loss(reconstruction[:, :, :COUNT_DIM], target[:, :, :COUNT_DIM])
    sequence_loss = torch.nn.functional.mse_loss(
        reconstruction[:, :, COUNT_DIM:], target[:, :, COUNT_DIM:].detach()
    )
    if view_mode == "count":
        return count_loss
    if view_mode == "sequence":
        return sequence_loss
    return 0.5 * count_loss + 0.5 * sequence_loss


def train_model(dataset: LazyCalendarWindowDataset, train_indices: np.ndarray, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, "num_workers", 0),
    )
    model = MultiViewTCNTransformerAutoencoder(**model_config(args)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    print(
        f"[TRAIN] device={device} windows={len(train_indices):,} batches/epoch={len(loader):,} "
        f"lazy_day_storage={dataset.storage_bytes / (1024 ** 2):.1f} MiB",
        flush=True,
    )
    for epoch in range(args.epochs):
        epoch_started = time.time()
        total = 0.0
        batches = 0
        model.train()
        for count, tokens, sources, gaps in loader:
            count, tokens, sources, gaps = count.to(device), tokens.to(device), sources.to(device), gaps.to(device)
            optimizer.zero_grad()
            reconstruction, target = model(count, tokens, sources, gaps)
            loss = reconstruction_loss(reconstruction, target, args.view_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            batches += 1
            progress_every = max(0, int(getattr(args, "progress_every", 250)))
            if progress_every and batches % progress_every == 0:
                print(
                    f"[TRAIN] epoch={epoch + 1}/{args.epochs} batch={batches:,}/{len(loader):,} "
                    f"mean_loss={total / batches:.6f}",
                    flush=True,
                )
        epoch_loss = total / max(batches, 1)
        losses.append(epoch_loss)
        print(
            f"[TRAIN] epoch={epoch + 1}/{args.epochs} complete "
            f"loss={epoch_loss:.6f} seconds={time.time() - epoch_started:.1f}",
            flush=True,
        )
    return model, losses


def score_model(
    model,
    dataset: LazyCalendarWindowDataset,
    batch_size: int,
    progress_every: int = 250,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    scores, count_errors, sequence_errors = [], [], []
    model.eval()
    print(f"[SCORE] windows={len(dataset):,} batches={len(loader):,}", flush=True)
    with torch.no_grad():
        for batch_index, (count, tokens, sources, gaps) in enumerate(loader, start=1):
            count, tokens, sources, gaps = count.to(device), tokens.to(device), sources.to(device), gaps.to(device)
            reconstruction, target = model(count, tokens, sources, gaps)
            count_error = torch.mean((reconstruction[:, -1, :COUNT_DIM] - target[:, -1, :COUNT_DIM]) ** 2, dim=1)
            sequence_error = torch.mean((reconstruction[:, -1, COUNT_DIM:] - target[:, -1, COUNT_DIM:]) ** 2, dim=1)
            score = 0.5 * count_error + 0.5 * sequence_error
            scores.append(score.cpu().numpy())
            count_errors.append(count_error.cpu().numpy())
            sequence_errors.append(sequence_error.cpu().numpy())
            if progress_every and batch_index % progress_every == 0:
                print(f"[SCORE] batch={batch_index:,}/{len(loader):,}", flush=True)
    return np.concatenate(scores), np.concatenate(count_errors), np.concatenate(sequence_errors)


def build_dataset(multiview: pd.DataFrame, count_projection: np.ndarray, args):
    count_columns = [f"z_count_{index}" for index in range(COUNT_DIM)]
    for index, column in enumerate(count_columns):
        multiview[column] = count_projection[:, index]
    print(
        f"[DATA] building compact calendar-day storage; max_events_per_day={args.max_events_per_day}",
        flush=True,
    )
    count, tokens, sources, gaps, slices, metas = make_multiview_day_inputs(
        multiview,
        count_columns,
        args.window_size,
        max_events_per_day=args.max_events_per_day,
        allow_padding=True,
    )
    dataset = LazyCalendarWindowDataset(
        count,
        tokens,
        sources,
        gaps,
        slices,
        args.window_size,
    )
    print(
        f"[DATA] calendar_days={len(count):,} windows={len(dataset):,} "
        f"storage={dataset.storage_bytes / (1024 ** 2):.1f} MiB",
        flush=True,
    )
    return dataset, metas


def fit_group_thresholds(score_frame: pd.DataFrame, args, column: str, level: str) -> dict:
    thresholds = {}
    for group_name, group in score_frame.groupby(column):
        if not str(group_name).strip() or str(group_name) == "UNKNOWN":
            continue
        enough_users = group["user"].nunique() >= args.min_role_users
        enough_user_days = len(group) >= args.min_role_user_days
        if enough_users or enough_user_days:
            thresholds[str(group_name)] = {
                "threshold": float(np.quantile(group["score"], args.anomaly_quantile)),
                "n_samples": int(len(group)),
                "n_users": int(group["user"].nunique()),
                "backoff_level": level,
            }
    return thresholds


def train_global(multiview: pd.DataFrame, active_days: int, model_dir: Path, args, started: float):
    multiview = assign_temporal_splits(multiview, args.train_fraction, args.validation_fraction)
    count_projection, vectorizer = fit_or_transform_count_view(multiview, args)
    dataset, metas = build_dataset(multiview, count_projection, args)
    if len(metas) == 0:
        raise ValueError("No calendar windows generated")
    meta_frame = pd.DataFrame(metas)
    train_mask = meta_frame["split"].eq("train").to_numpy()
    training_label_policy = getattr(args, "training_label_policy", "ignore")
    if training_label_policy == "exclude-positive-windows":
        train_mask &= ~meta_frame["window_has_positive"].astype(bool).to_numpy()
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(meta_frame["split"].eq("validation").to_numpy())
    if len(train_indices) == 0:
        raise ValueError("No train windows generated")
    threshold_source = "validation"
    if len(validation_indices) == 0:
        validation_indices = train_indices
        threshold_source = "train_fallback_no_validation_windows"

    model, losses = train_model(dataset, train_indices, args)
    scores, count_errors, sequence_errors = score_model(
        model,
        dataset,
        args.batch_size,
        progress_every=getattr(args, "progress_every", 250),
        num_workers=getattr(args, "num_workers", 0),
    )
    score_frame = meta_frame.copy()
    score_frame["score"] = scores
    score_frame["e_count"] = count_errors
    score_frame["e_seq"] = sequence_errors
    calibration_label_policy = getattr(args, "calibration_label_policy", "ignore")
    if calibration_label_policy == "benign-only":
        benign_validation = meta_frame.iloc[validation_indices]["label_day"].astype(int).eq(0).to_numpy()
        validation_indices = validation_indices[benign_validation]
        if len(validation_indices) == 0:
            raise ValueError("No benign validation windows available for threshold calibration")
        threshold_source += "_benign_only"
    validation_scores = scores[validation_indices]
    threshold = float(np.quantile(validation_scores, args.anomaly_quantile))
    validation_frame = score_frame.iloc[validation_indices]
    role_thresholds = fit_group_thresholds(validation_frame, args, "role", "role")
    department_thresholds = fit_group_thresholds(validation_frame, args, "department", "department")

    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    torch.save(
        {
            "architecture": ARCHITECTURE,
            "model_state": model.state_dict(),
            "model_config": model_config(args),
        },
        model_dir / "model.pt",
    )
    metadata = {
        "scope": "global",
        "architecture": ARCHITECTURE,
        "active_days": active_days,
        "windows": int(len(score_frame)),
        "split_windows": score_frame["split"].value_counts().to_dict(),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_method": "quantile",
        "anomaly_quantile": args.anomaly_quantile,
        "score_formula": "e_total = 0.5 * e_count + 0.5 * e_seq",
        "score_weights": {"count": 0.5, "sequence": 0.5, "context": 0.0},
        "sequence_representation": "raw token/source/time-gap neural encoder",
        "context_reconstruction_enabled": False,
        "calendar_windowing": True,
        "training_mode": (
            "clean_one_class_label_filtered"
            if training_label_policy == "exclude-positive-windows"
            else "contaminated_unsupervised"
        ),
        "training_label_policy": training_label_policy,
        "calibration_label_policy": calibration_label_policy,
        "labels_used_for_fit": training_label_policy != "ignore" or calibration_label_policy != "ignore",
        "lazy_window_loading": True,
        "compact_day_storage_mib": round(dataset.storage_bytes / (1024 ** 2), 3),
        "max_events_per_day": args.max_events_per_day,
        "truncated_endpoint_days": int(meta_frame["sequence_was_truncated"].sum()),
        "truncated_endpoint_fraction": float(meta_frame["sequence_was_truncated"].mean()),
        "random_seed": getattr(args, "random_seed", 42),
        "role_threshold_minimum": {
            "min_users": args.min_role_users,
            "min_user_days": args.min_role_user_days,
            "rule": "min_users OR min_user_days",
        },
        "role_thresholds": role_thresholds,
        "department_thresholds": department_thresholds,
        "losses": losses,
        "model_config": model_config(args),
        "model_version": f"global-role-{int(time.time())}",
        "training_seconds": round(time.time() - started, 3),
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    score_frame.to_csv(model_dir / "all_scores.csv", index=False)
    return {
        "ok": True,
        "eligible": True,
        "updated": True,
        "scope": "global",
        "user_id": None,
        "model_dir": str(model_dir),
        "windows": int(len(score_frame)),
        "events": 0,
        "active_days": active_days,
        "safe_days": 0,
        "threshold": threshold,
        "message": "global multi-view token-encoder model trained",
    }


def load_global_model(global_dir: Path):
    checkpoint = torch.load(global_dir / "model.pt", map_location="cpu")
    if checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Personal threshold calibration requires a retrained multiview-token-encoder-v1 global model")
    model = MultiViewTCNTransformerAutoencoder(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    metadata = json.loads((global_dir / "metadata.json").read_text(encoding="utf-8"))
    return model, metadata, checkpoint["model_config"]


def group_threshold(global_metadata: dict, role: str, department: str) -> tuple[float, str]:
    global_threshold = float(global_metadata["threshold"])
    role_info = global_metadata.get("role_thresholds", {}).get(role)
    department_info = global_metadata.get("department_thresholds", {}).get(department)
    if role_info:
        return float(role_info["threshold"]), "role"
    if department_info:
        return float(department_info["threshold"]), "department"
    return global_threshold, "global"


def calibrate_personal(multiview: pd.DataFrame, active_days: int, model_dir: Path, args):
    global_dir = Path(args.global_model_dir or "../models/global")
    model, global_metadata, global_model_config = load_global_model(global_dir)
    # The global model dictates event truncation and architecture at calibration time.
    args.max_events_per_day = int(global_model_config["max_events_per_day"])
    args.window_size = int(global_model_config["window_size"])
    args.view_mode = str(global_model_config.get("view_mode", "count-sequence"))
    count_projection, _ = fit_or_transform_count_view(multiview, args)
    dataset, metas = build_dataset(multiview, count_projection, args)
    if not metas:
        raise ValueError("No personalized calendar windows generated")
    scores, count_errors, sequence_errors = score_model(
        model,
        dataset,
        args.batch_size,
        progress_every=getattr(args, "progress_every", 250),
        num_workers=getattr(args, "num_workers", 0),
    )
    score_frame = pd.DataFrame(metas)
    score_frame["date"] = pd.to_datetime(score_frame["date"], errors="coerce")
    score_frame["score"] = scores
    score_frame["e_count"] = count_errors
    score_frame["e_seq"] = sequence_errors
    score_frame = score_frame.sort_values("date").reset_index(drop=True)
    latest = score_frame.iloc[-1]
    role = str(latest.get("role") or "UNKNOWN")
    department = str(latest.get("department") or "UNKNOWN")
    role_threshold, role_backoff = group_threshold(global_metadata, role, department)

    last_observed_date = score_frame["date"].max()
    safe_cutoff = last_observed_date - pd.Timedelta(days=args.update_delay_days)
    history_start = safe_cutoff - pd.Timedelta(days=args.safe_history_days - 1)
    role_alert = score_frame["score"] >= role_threshold
    after_alert = role_alert.shift(1, fill_value=False)
    score_frame["is_safe"] = (
        (score_frame["date"] >= history_start)
        & (score_frame["date"] <= safe_cutoff)
        & ~role_alert
        & ~after_alert
    )
    safe_scores = score_frame.loc[score_frame["is_safe"], "score"].to_numpy(dtype=float)
    safe_days = int(len(safe_scores))
    eligible = safe_days >= args.min_safe_days
    metadata_path = model_dir / "metadata.json"
    old_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    old_safe_days = int(old_metadata.get("safe_days_count", 0))
    old_last_scored = pd.to_datetime(old_metadata.get("last_scored_date"), errors="coerce")
    calendar_days_since_update = (
        int((last_observed_date - old_last_scored).days) if pd.notna(old_last_scored) else args.update_every_calendar_days
    )
    due = (
        args.force_update
        or not old_metadata
        or safe_days - old_safe_days >= args.update_every_safe_days
        or calendar_days_since_update >= args.update_every_calendar_days
    )
    if not eligible:
        pending_metadata = {
            "scope": "personalized",
            "calibration_type": "threshold_only_no_personal_neural_model",
            "is_active": False,
            "user_id": args.user_id,
            "threshold": float(old_metadata.get("threshold", 0.0)),
            "safe_days_count": safe_days,
            "active_days": active_days,
            "safe_history_days": args.safe_history_days,
            "update_delay_days": args.update_delay_days,
            "last_scored_date": last_observed_date.strftime("%Y-%m-%d"),
            "source_global_model_version": global_metadata.get("model_version"),
        }
        metadata_path.write_text(json.dumps(pending_metadata, indent=2), encoding="utf-8")
        result = {
            "ok": True,
            "eligible": False,
            "updated": False,
            "scope": "personalized",
            "user_id": args.user_id,
            "model_dir": str(model_dir),
            "windows": int(len(score_frame)),
            "events": 0,
            "active_days": active_days,
            "safe_days": safe_days,
            "threshold": float(old_metadata.get("threshold", 0.0)),
            "message": f"Personal threshold not activated: {safe_days} safe days; need {args.min_safe_days}.",
        }
        score_frame.to_csv(model_dir / "calibration_scores.csv", index=False)
        return result
    if not due:
        return {
            "ok": True,
            "eligible": True,
            "updated": False,
            "scope": "personalized",
            "user_id": args.user_id,
            "model_dir": str(model_dir),
            "windows": int(len(score_frame)),
            "events": 0,
            "active_days": active_days,
            "safe_days": safe_days,
            "threshold": float(old_metadata["threshold"]),
            "message": "Personal threshold is current; update cadence has not been reached.",
        }

    median = float(np.median(safe_scores))
    q1, q3 = np.quantile(safe_scores, [0.25, 0.75])
    iqr = float(q3 - q1)
    threshold = median + 3.0 * iqr
    method = "median_plus_3_iqr"
    if not np.isfinite(threshold) or threshold <= 0 or iqr == 0:
        threshold = float(np.quantile(safe_scores, 0.99))
        method = "quantile_0.99_fallback"
    metadata = {
        "scope": "personalized",
        "calibration_type": "threshold_only_no_personal_neural_model",
        "is_active": True,
        "user_id": args.user_id,
        "threshold": threshold,
        "threshold_method": method,
        "median_safe": median,
        "iqr_safe": iqr,
        "safe_days_count": safe_days,
        "active_days": active_days,
        "safe_history_days": args.safe_history_days,
        "update_delay_days": args.update_delay_days,
        "update_every_safe_days": args.update_every_safe_days,
        "update_every_calendar_days": args.update_every_calendar_days,
        "last_safe_date": score_frame.loc[score_frame["is_safe"], "date"].max().strftime("%Y-%m-%d"),
        "last_scored_date": last_observed_date.strftime("%Y-%m-%d"),
        "role": role,
        "department": department,
        "role_backoff": role_backoff,
        "source_global_model_version": global_metadata.get("model_version"),
        "model_version": f"personal-threshold-{args.user_id}-{int(time.time())}",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    score_frame.to_csv(model_dir / "calibration_scores.csv", index=False)
    return {
        "ok": True,
        "eligible": True,
        "updated": True,
        "scope": "personalized",
        "user_id": args.user_id,
        "model_dir": str(model_dir),
        "windows": int(len(score_frame)),
        "events": 0,
        "active_days": active_days,
        "safe_days": safe_days,
        "threshold": threshold,
        "message": "safe personalized threshold updated; the global neural model remains unchanged",
    }


def main():
    args = parse_args()
    started = time.time()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
    configure_feature_rules(args.feature_rules)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    multiview, active_days = load_multiview(args)
    if multiview.empty:
        raise ValueError("No multiview rows available")
    multiview = calendarize_multiview(multiview)
    if args.scope == "global":
        result = train_global(multiview, active_days, model_dir, args, started)
    else:
        result = calibrate_personal(multiview, active_days, model_dir, args)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
