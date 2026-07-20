from __future__ import annotations
import argparse, json, time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from model import TCNTransformerAutoencoder
from multiview_features import (
    COUNT_FEATURE_COLUMNS, SEQ_FEATURE_COLUMNS, load_ldap_context, read_events_from_sqlite,
    multiview_from_events, make_windows_from_multiview
)


def safe_signed_log_matrix(mv: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Convert feature columns to a finite numeric matrix and compress scale safely.

    Some count-view columns are true counts, but deviation columns such as
    z_file_vs_user_30d can be negative. Plain np.log1p(x) is invalid when
    x < -1, so use signed log1p: sign(x) * log1p(abs(x)).
    """
    if not cols:
        return np.zeros((len(mv), 0), dtype=np.float64)
    work = mv[cols].copy()
    for c in cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    arr = work.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float64)
    arr = np.clip(arr, -1e12, 1e12)
    arr = np.sign(arr) * np.log1p(np.abs(arr))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scope", choices=["global", "personalized"], required=True)
    p.add_argument("--multiview-csv")
    p.add_argument("--db")
    p.add_argument("--user-id")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--global-model-dir")
    p.add_argument("--role-context")
    p.add_argument("--window-size", type=int, default=30)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-active-days", type=int, default=30)
    p.add_argument("--anomaly-quantile", type=float, default=0.995)
    p.add_argument("--min-role-samples", type=int, default=25)
    return p.parse_args()

def load_multiview(args):
    if args.scope == "global":
        if not args.multiview_csv: raise ValueError("--multiview-csv is required")
        path = Path(args.multiview_csv)
        if not path.exists(): raise FileNotFoundError(f"Multiview file not found: {path}")
        return pd.read_csv(path)
    role_context = load_ldap_context(Path(args.role_context).parent if args.role_context else None)
    events = read_events_from_sqlite(args.db, user_id=args.user_id, role_context=role_context)
    mv = multiview_from_events(events)
    active_days = mv["date"].nunique() if not mv.empty else 0
    if active_days < args.min_active_days:
        raise ValueError(f"User {args.user_id} has {active_days} active days; need >= {args.min_active_days}")
    return mv

def fit_or_transform_views(mv, args):
    count_cols = COUNT_FEATURE_COLUMNS
    seq_cols = SEQ_FEATURE_COLUMNS
    for c in count_cols + seq_cols:
        if c not in mv.columns: mv[c] = 0
    if args.scope == "global":
        count_scaler = RobustScaler()
        seq_scaler = RobustScaler()
        count_raw = safe_signed_log_matrix(mv, count_cols)
        seq_raw = safe_signed_log_matrix(mv, seq_cols)
        count_scaled = count_scaler.fit_transform(count_raw)
        seq_scaled = seq_scaler.fit_transform(seq_raw)
        count_pca = PCA(n_components=64, random_state=42)
        seq_pca = PCA(n_components=64, random_state=42)
        # If columns/samples are less than 64, pad after PCA.
        count_z = fit_pca_64(count_pca, count_scaled)
        seq_z = fit_pca_64(seq_pca, seq_scaled)
        return count_z, seq_z, {"count_scaler": count_scaler, "seq_scaler": seq_scaler, "count_pca": count_pca, "seq_pca": seq_pca, "count_cols": count_cols, "seq_cols": seq_cols}
    global_dir = Path(args.global_model_dir or "../models/global")
    vectorizer = joblib.load(global_dir / "vectorizer.joblib")
    count_raw = safe_signed_log_matrix(mv, vectorizer["count_cols"])
    seq_raw = safe_signed_log_matrix(mv, vectorizer["seq_cols"])
    count_scaled = vectorizer["count_scaler"].transform(count_raw)
    seq_scaled = vectorizer["seq_scaler"].transform(seq_raw)
    count_z = transform_pca_64(vectorizer["count_pca"], count_scaled)
    seq_z = transform_pca_64(vectorizer["seq_pca"], seq_scaled)
    return count_z, seq_z, vectorizer

def fit_pca_64(pca, x):
    n_comp = min(64, x.shape[0], x.shape[1])
    pca.n_components = n_comp
    z = pca.fit_transform(x)
    return pad64(z)

def transform_pca_64(pca, x):
    z = pca.transform(x)
    return pad64(z)

def pad64(z):
    if z.shape[1] < 64:
        return np.hstack([z, np.zeros((z.shape[0], 64-z.shape[1]), dtype=z.dtype)])
    return z[:, :64]

def train_model(x, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = TCNTransformerAutoencoder(input_dim=x.shape[-1], window_size=x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses=[]
    for _ in range(args.epochs):
        total=0; batches=0
        for (b,) in loader:
            b=b.to(device)
            opt.zero_grad()
            r=model(b)
            loss=torch.nn.functional.mse_loss(r,b)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total += float(loss.item()); batches += 1
        losses.append(total/max(batches,1))
    model.eval()
    with torch.no_grad():
        recon = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    count_err = np.mean((recon[:, -1, :64] - x[:, -1, :64]) ** 2, axis=1)
    seq_err = np.mean((recon[:, -1, 64:] - x[:, -1, 64:]) ** 2, axis=1)
    score = 0.5 * count_err + 0.5 * seq_err
    return model, losses, score, count_err, seq_err

def main():
    args=parse_args(); started=time.time()
    model_dir=Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    mv=load_multiview(args).copy()
    count_z, seq_z, vectorizer = fit_or_transform_views(mv, args)
    mv["role"] = mv.get("role", "UNKNOWN").fillna("UNKNOWN").replace("", "UNKNOWN")
    mv["department"] = mv.get("department", "UNKNOWN").fillna("UNKNOWN").replace("", "UNKNOWN")
    z=np.hstack([count_z, seq_z]).astype(np.float32)
    feature_cols=[f"z{i}" for i in range(128)]
    for i,c in enumerate(feature_cols): mv[c]=z[:,i]
    x, metas = make_windows_from_multiview(mv, feature_cols, args.window_size, allow_padding=True)
    if len(x)==0: raise ValueError("No windows generated")
    model, losses, scores, count_err, seq_err = train_model(x, args)
    tau = float(np.quantile(scores, args.anomaly_quantile))
    role_thresholds = {}
    dept_thresholds = {}
    if args.scope == "global":
        score_df=pd.DataFrame(metas)
        score_df["score"]=scores
        for role,g in score_df.groupby("role"):
            if len(g)>=args.min_role_samples:
                role_thresholds[str(role)]={"threshold": float(np.quantile(g["score"], args.anomaly_quantile)), "n_samples": int(len(g)), "backoff_level":"role"}
        for dep,g in score_df.groupby("department"):
            if len(g)>=args.min_role_samples:
                dept_thresholds[str(dep)]={"threshold": float(np.quantile(g["score"], args.anomaly_quantile)), "n_samples": int(len(g)), "backoff_level":"department"}
        joblib.dump(vectorizer, model_dir/"vectorizer.joblib")
    torch.save({"model_state": model.state_dict(), "input_dim": x.shape[-1], "window_size": x.shape[1]}, model_dir/"model.pt")
    metadata={
        "scope": args.scope, "user_id": args.user_id, "active_days": int(mv["date"].nunique()), "windows": int(len(x)),
        "threshold": tau, "threshold_method": "quantile", "anomaly_quantile": args.anomaly_quantile,
        "score_formula": "e_total = 0.5 * e_count + 0.5 * e_seq",
        "score_weights": {"count": 0.5, "sequence": 0.5, "context": 0.0},
        "context_reconstruction_enabled": False,
        "feature_transform": "safe_signed_log1p = sign(x) * log1p(abs(x))",
        "role_thresholds": role_thresholds, "department_thresholds": dept_thresholds,
        "losses": losses, "model_version": f"{args.scope}-{args.user_id or 'role-global'}-{int(time.time())}",
        "training_seconds": round(time.time()-started,3), "score_mean": float(np.mean(scores)), "score_median": float(np.median(scores))
    }
    (model_dir/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    pd.DataFrame(metas).assign(score=scores, e_count=count_err, e_seq=seq_err).to_csv(model_dir/"train_scores.csv", index=False)
    print(json.dumps({"ok": True, "scope": args.scope, "user_id": args.user_id, "model_dir": str(model_dir), "windows": int(len(x)), "events": 0, "active_days": int(mv["date"].nunique()), "threshold": tau, "message": f"{args.scope} multi-view model trained"}))
if __name__ == "__main__": main()
