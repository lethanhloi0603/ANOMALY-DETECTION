from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from model import MultiViewTCNTransformerAutoencoder
from multiview_features import (
    configure_feature_rules,
    load_ldap_context,
    make_multiview_model_inputs,
    multiview_from_events,
    read_events_from_sqlite,
)


ARCHITECTURE = "multiview-token-encoder-v1"
COUNT_DIM = 64


def safe_signed_log_matrix(multiview: pd.DataFrame, columns: list[str]) -> np.ndarray:
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
    parser.add_argument("--db", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--scope", choices=["global", "personalized"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--global-model-dir", required=True)
    parser.add_argument("--role-context")
    parser.add_argument("--feature-rules")
    parser.add_argument("--window-size", type=int, default=30)
    return parser.parse_args()


def warn(message: str):
    print(json.dumps({
        "score": 0.0,
        "threshold": 0.0,
        "is_anomaly": False,
        "model_version": "no_model",
        "warning": message,
    }))


def pad64(values: np.ndarray) -> np.ndarray:
    if values.shape[1] < COUNT_DIM:
        values = np.hstack([values, np.zeros((values.shape[0], COUNT_DIM - values.shape[1]), dtype=values.dtype)])
    return values[:, :COUNT_DIM]


def vectorize_count(multiview: pd.DataFrame, vectorizer: dict) -> np.ndarray:
    columns = vectorizer["count_cols"]
    for column in columns:
        if column not in multiview.columns:
            multiview[column] = 0
    raw = safe_signed_log_matrix(multiview, columns)
    scaled = vectorizer["count_scaler"].transform(raw)
    return pad64(vectorizer["count_pca"].transform(scaled)).astype(np.float32)


def score_new_model(multiview: pd.DataFrame, checkpoint: dict, vectorizer: dict):
    config = checkpoint["model_config"]
    count_projection = vectorize_count(multiview, vectorizer)
    count_columns = [f"z_count_{index}" for index in range(COUNT_DIM)]
    for index, column in enumerate(count_columns):
        multiview[column] = count_projection[:, index]
    count, tokens, sources, gaps, metas = make_multiview_model_inputs(
        multiview,
        count_columns,
        int(config["window_size"]),
        max_events_per_day=int(config["max_events_per_day"]),
        allow_padding=True,
    )
    if not metas:
        raise ValueError("No calendar window generated")
    model = MultiViewTCNTransformerAutoencoder(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        reconstruction, target = model(
            torch.tensor(count[-1:], dtype=torch.float32),
            torch.tensor(tokens[-1:], dtype=torch.long),
            torch.tensor(sources[-1:], dtype=torch.long),
            torch.tensor(gaps[-1:], dtype=torch.float32),
        )
    count_error = float(torch.mean((reconstruction[0, -1, :COUNT_DIM] - target[0, -1, :COUNT_DIM]) ** 2))
    sequence_error = float(torch.mean((reconstruction[0, -1, COUNT_DIM:] - target[0, -1, COUNT_DIM:]) ** 2))
    return 0.5 * count_error + 0.5 * sequence_error, count_error, sequence_error, metas[-1]


def main():
    args = parse_args()
    configure_feature_rules(args.feature_rules)
    model_dir = Path(args.model_dir)
    global_dir = Path(args.global_model_dir)
    if not (global_dir / "model.pt").exists() or not (global_dir / "vectorizer.joblib").exists():
        warn("Global model/vectorizer not found. Train global first.")
        return

    role_context = load_ldap_context(args.role_context)
    events = read_events_from_sqlite(args.db, user_id=args.user_id, role_context=role_context)
    if events.empty:
        warn(f"No logs found for user {args.user_id}")
        return
    multiview = multiview_from_events(events)
    if multiview.empty:
        warn(f"No user-day multiview rows for {args.user_id}")
        return

    checkpoint = torch.load(global_dir / "model.pt", map_location="cpu")
    vectorizer = joblib.load(global_dir / "vectorizer.joblib")
    if checkpoint.get("architecture") != ARCHITECTURE:
        warn("Legacy global model detected. Retrain global after regenerating calendar-day multiview data; legacy active-day/sequence-derived scores are intentionally rejected.")
        return
    score, count_error, sequence_error, latest_meta = score_new_model(multiview, checkpoint, vectorizer)

    global_metadata = json.loads((global_dir / "metadata.json").read_text(encoding="utf-8"))
    global_threshold = float(global_metadata.get("threshold", 1e-9)) or 1e-9
    role = str(latest_meta.get("role") or "UNKNOWN")
    department = str(latest_meta.get("department") or "UNKNOWN")
    role_info = global_metadata.get("role_thresholds", {}).get(role)
    department_info = global_metadata.get("department_thresholds", {}).get(department)
    if role_info:
        role_threshold = float(role_info["threshold"])
        backoff = "role"
    elif department_info:
        role_threshold = float(department_info["threshold"])
        backoff = "department"
    else:
        role_threshold = global_threshold
        backoff = "global"

    global_ratio = score / global_threshold if global_threshold > 0 else 0.0
    role_ratio = score / role_threshold if role_threshold > 0 else global_ratio
    personal_ratio = None
    warning_parts = []
    threshold = role_threshold
    final_index = 0.40 * global_ratio + 0.60 * role_ratio if backoff != "global" else global_ratio
    model_version = global_metadata.get("model_version")

    if args.scope == "personalized":
        personal_metadata_path = model_dir / "metadata.json"
        if personal_metadata_path.exists():
            personal_metadata = json.loads(personal_metadata_path.read_text(encoding="utf-8"))
            source_version = personal_metadata.get("source_global_model_version")
            if source_version and source_version != global_metadata.get("model_version"):
                warning_parts.append("Personal threshold is stale after a global model change; using role/global calibration.")
            elif personal_metadata.get("calibration_type") != "threshold_only_no_personal_neural_model":
                warning_parts.append("Legacy personalized neural model ignored; update the safe personal threshold.")
            elif not personal_metadata.get("is_active", True):
                warning_parts.append("Personal threshold is pending enough delayed safe days; using role/global calibration.")
            else:
                personal_threshold = float(personal_metadata.get("threshold", 0.0))
                if personal_threshold > 0:
                    personal_ratio = score / personal_threshold
                    final_index = 0.25 * global_ratio + 0.35 * role_ratio + 0.40 * personal_ratio
                    threshold = personal_threshold
                    model_version = f"{model_version}+{personal_metadata.get('model_version')}"
        else:
            warning_parts.append("Personal threshold metadata not found; using role/global calibration.")
    if backoff == "global":
        warning_parts.append("Role threshold missing or under-sampled; backed off to global threshold.")

    print(json.dumps({
        "score": score,
        "e_count": count_error,
        "e_seq": sequence_error,
        "threshold": threshold,
        "is_anomaly": bool(final_index > 1.0),
        "model_version": model_version,
        "role": role,
        "department": department,
        "global_ratio": global_ratio,
        "role_ratio": role_ratio,
        "personal_ratio": personal_ratio,
        "final_anomaly_index": final_index,
        "warning": " ".join(warning_parts) or None,
    }))


if __name__ == "__main__":
    main()
