from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from features import latest_user_window, load_feature_columns, read_events_from_sqlite
from model import TCNTransformerAutoencoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--user-id", required=True)
    p.add_argument("--scope", choices=["global", "personalized"], required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--window-size", type=int, default=30)
    return p.parse_args()


def print_fallback(reason: str):
    print(json.dumps({"score": 0.0, "threshold": 0.0, "is_anomaly": False, "model_version": "heuristic_no_model", "warning": reason}))


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    model_path = model_dir / "model.pt"
    scaler_path = model_dir / "scaler.joblib"
    feature_path = model_dir / "feature_columns.json"
    metadata_path = model_dir / "metadata.json"
    if not model_path.exists() or not scaler_path.exists() or not feature_path.exists():
        print_fallback(f"Model files not found for scope={args.scope}. Train model first. Expected {model_dir}")
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    feature_columns = load_feature_columns(feature_path)
    scaler = joblib.load(scaler_path)
    events = read_events_from_sqlite(args.db, user_id=args.user_id)
    if events.empty:
        print_fallback(f"No logs found for user {args.user_id}")
        return

    x = latest_user_window(events, args.window_size, feature_columns)
    n, w, d = x.shape
    x_scaled = scaler.transform(x.reshape(-1, d)).reshape(n, w, d).astype(np.float32)
    ckpt = torch.load(model_path, map_location="cpu")
    model = TCNTransformerAutoencoder(input_dim=int(ckpt["input_dim"]), window_size=int(ckpt["window_size"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(x_scaled, dtype=torch.float32)).numpy()
    score = float(np.mean((recon[0, -1, :] - x_scaled[0, -1, :]) ** 2))
    threshold = float(metadata.get("threshold", 0.0))
    print(json.dumps({"score": score, "threshold": threshold, "is_anomaly": bool(score > threshold) if threshold > 0 else False, "model_version": metadata.get("model_version", f"{args.scope}-unknown"), "warning": None}))


if __name__ == "__main__":
    main()
