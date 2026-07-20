from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from model import TCNTransformerAutoencoder
from multiview_features import read_events_from_sqlite, multiview_from_events, make_windows_from_multiview, load_ldap_context


def safe_signed_log_matrix(mv: pd.DataFrame, cols: list[str]) -> np.ndarray:
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
    p=argparse.ArgumentParser()
    p.add_argument("--db", required=True); p.add_argument("--user-id", required=True)
    p.add_argument("--scope", choices=["global","personalized"], required=True)
    p.add_argument("--model-dir", required=True); p.add_argument("--global-model-dir", required=True)
    p.add_argument("--role-context"); p.add_argument("--window-size", type=int, default=30)
    return p.parse_args()

def warn(msg):
    print(json.dumps({"score":0.0,"threshold":0.0,"is_anomaly":False,"model_version":"no_model","warning":msg}))

def pad64(z):
    if z.shape[1] < 64: return np.hstack([z, np.zeros((z.shape[0],64-z.shape[1]), dtype=z.dtype)])
    return z[:,:64]

def vectorize(mv, global_dir):
    vec=joblib.load(Path(global_dir)/"vectorizer.joblib")
    for c in vec["count_cols"]+vec["seq_cols"]:
        if c not in mv.columns: mv[c]=0
    count_raw=safe_signed_log_matrix(mv, vec["count_cols"])
    seq_raw=safe_signed_log_matrix(mv, vec["seq_cols"])
    count_z=pad64(vec["count_pca"].transform(vec["count_scaler"].transform(count_raw)))
    seq_z=pad64(vec["seq_pca"].transform(vec["seq_scaler"].transform(seq_raw)))
    return np.hstack([count_z, seq_z]).astype(np.float32)

def load_model(model_dir):
    ckpt=torch.load(Path(model_dir)/"model.pt", map_location="cpu")
    model=TCNTransformerAutoencoder(input_dim=int(ckpt["input_dim"]), window_size=int(ckpt["window_size"]))
    model.load_state_dict(ckpt["model_state"]); model.eval()
    meta=json.loads((Path(model_dir)/"metadata.json").read_text(encoding="utf-8"))
    return model, meta

def main():
    args=parse_args(); model_dir=Path(args.model_dir); global_dir=Path(args.global_model_dir)
    if not (global_dir/"model.pt").exists() or not (global_dir/"vectorizer.joblib").exists():
        warn("Global model/vectorizer not found. Train global first."); return
    if args.scope=="personalized" and not (model_dir/"model.pt").exists():
        warn(f"Personalized model not found: {model_dir}"); return
    selected_dir = model_dir if args.scope=="personalized" else global_dir
    role_context=load_ldap_context(Path(args.role_context).parent if args.role_context else None)
    events=read_events_from_sqlite(args.db, user_id=args.user_id, role_context=role_context)
    if events.empty:
        warn(f"No logs found for user {args.user_id}"); return
    mv=multiview_from_events(events)
    if mv.empty:
        warn(f"No user-day multiview rows for {args.user_id}"); return
    z=vectorize(mv, global_dir)
    feature_cols=[f"z{i}" for i in range(128)]
    for i,c in enumerate(feature_cols): mv[c]=z[:,i]
    x, metas=make_windows_from_multiview(mv, feature_cols, args.window_size, allow_padding=True)
    if len(x)==0:
        warn("No window generated"); return
    model, meta=load_model(selected_dir)
    with torch.no_grad(): recon=model(torch.tensor(x[-1:], dtype=torch.float32)).numpy()
    latest=x[-1:]
    e_count=float(np.mean((recon[0,-1,:64]-latest[0,-1,:64])**2))
    e_seq=float(np.mean((recon[0,-1,64:]-latest[0,-1,64:])**2))
    score=0.5*e_count+0.5*e_seq
    latest_meta=metas[-1] if metas else {"role":"UNKNOWN","department":"UNKNOWN"}
    role=str(latest_meta.get("role") or "UNKNOWN"); dep=str(latest_meta.get("department") or "UNKNOWN")
    global_meta=json.loads((global_dir/"metadata.json").read_text(encoding="utf-8"))
    tau_global=float(global_meta.get("threshold", 1e-9)) or 1e-9
    role_info=global_meta.get("role_thresholds",{}).get(role)
    dep_info=global_meta.get("department_thresholds",{}).get(dep)
    if role_info: tau_role=float(role_info["threshold"]); backoff="role"
    elif dep_info: tau_role=float(dep_info["threshold"]); backoff="department"
    else: tau_role=tau_global; backoff="global"
    global_ratio=score/tau_global if tau_global>0 else 0
    role_ratio=score/tau_role if tau_role>0 else global_ratio
    personal_ratio=None
    if args.scope=="personalized":
        tau_personal=float(meta.get("threshold", tau_role)) or tau_role
        personal_ratio=score/tau_personal if tau_personal>0 else role_ratio
        final_index=0.25*global_ratio+0.35*role_ratio+0.40*personal_ratio
        threshold=tau_personal
    else:
        final_index=0.40*global_ratio+0.60*role_ratio if backoff!="global" else global_ratio
        threshold=tau_role
    print(json.dumps({
        "score": score, "threshold": threshold, "is_anomaly": bool(final_index>1.0),
        "model_version": meta.get("model_version"), "role": role, "department": dep,
        "global_ratio": global_ratio, "role_ratio": role_ratio, "personal_ratio": personal_ratio,
        "final_anomaly_index": final_index,
        "warning": None if backoff!="global" else "Role threshold missing; backed off to global threshold"
    }))
if __name__=="__main__": main()
