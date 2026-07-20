from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from features import build_daily_features, make_windows, normalize_event_frame, read_events_from_sqlite, save_feature_columns
from model import TCNTransformerAutoencoder


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=["global", "personalized"], required=True)
    p.add_argument("--events-csv", default=None)
    p.add_argument("--db", default=None)
    p.add_argument("--user-id", default=None)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-events", type=int, default=50)
    p.add_argument("--anomaly-quantile", type=float, default=0.95)
    return p.parse_args()


def load_events(args):
    if args.scope == "global":
        if not args.events_csv:
            raise ValueError("--events-csv is required for global training")
        path = Path(args.events_csv)
        if not path.exists():
            raise FileNotFoundError(f"Global events CSV not found: {path}")
        return normalize_event_frame(pd.read_csv(path))
    if not args.db or not args.user_id:
        raise ValueError("--db and --user-id are required for personalized training")
    events = read_events_from_sqlite(args.db, user_id=args.user_id)
    if len(events) < args.min_events:
        raise ValueError(f"User {args.user_id} has {len(events)} logs; need >= {args.min_events}")
    return events


def main():
    args = parse_args()
    started = time.time()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    events = load_events(args)
    daily = build_daily_features(events)
    x, feature_columns = make_windows(daily, window_size=args.window_size, allow_padding=True)
    if len(x) == 0:
        raise ValueError("No windows generated")

    n, w, d = x.shape
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.reshape(-1, d)).reshape(n, w, d).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(torch.tensor(x_scaled, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = TCNTransformerAutoencoder(input_dim=d, window_size=w).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    model.train()
    losses = []
    for _ in range(args.epochs):
        epoch_loss = 0.0
        batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        losses.append(epoch_loss / max(batches, 1))

    model.eval()
    with torch.no_grad():
        all_x = torch.tensor(x_scaled, dtype=torch.float32, device=device)
        recon = model(all_x).cpu().numpy()
    mse_latest = np.mean((recon[:, -1, :] - x_scaled[:, -1, :]) ** 2, axis=1)
    threshold = float(np.quantile(mse_latest, args.anomaly_quantile))

    torch.save({"model_state": model.state_dict(), "input_dim": d, "window_size": w}, model_dir / "model.pt")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    save_feature_columns(model_dir / "feature_columns.json", feature_columns)
    metadata = {
        "scope": args.scope,
        "user_id": args.user_id,
        "events": int(len(events)),
        "daily_rows": int(len(daily)),
        "windows": int(len(x)),
        "feature_dim": int(d),
        "window_size": int(w),
        "threshold": threshold,
        "anomaly_quantile": args.anomaly_quantile,
        "epochs": args.epochs,
        "losses": losses,
        "training_seconds": round(time.time() - started, 3),
        "model_version": f"{args.scope}-{args.user_id or 'company'}-{int(time.time())}",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "scope": args.scope, "user_id": args.user_id, "model_dir": str(model_dir), "windows": int(len(x)), "events": int(len(events)), "threshold": threshold, "message": f"{args.scope} model trained successfully"}))


if __name__ == "__main__":
    main()
