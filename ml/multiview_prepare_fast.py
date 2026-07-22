from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from multiview_features import (
    attach_cert_labels,
    build_unified_event_log,
    build_user_day_features,
    build_user_day_sequences,
    build_user_day_multiview,
    configure_feature_rules,
)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--cert-dir", required=True)
    p.add_argument("--ldap-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--holdout-users", type=int, default=2)
    p.add_argument("--min-active-days", type=int, default=30)
    p.add_argument("--max-rows-per-file", type=int, default=200_000)
    p.add_argument("--sampling-mode", choices=["full", "time-user-stratified", "head"], default="time-user-stratified")
    p.add_argument(
        "--holdout-strategy",
        choices=["role-stratified", "role-stratified-anomaly-demo", "top-active"],
        default="role-stratified",
    )
    p.add_argument(
        "--output-profile",
        choices=["compact", "audit"],
        default="compact",
        help="compact writes only training/runtime artifacts; audit also keeps large intermediate CSV/JSONL files.",
    )
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--insiders-file")
    p.add_argument("--feature-rules")
    return p.parse_args()

def sanitize(value: str) -> str:
    invalid='<>:"/\\|?*'
    return ''.join('_' if c in invalid else c for c in str(value))


def choose_holdout_users(multiview: pd.DataFrame, count: int, minimum_active_days: int, strategy: str, seed: int) -> list[str]:
    candidate_frame = (
        multiview.groupby("user", as_index=False)
        .agg(
            active_days=("date", "nunique"),
            role=("role", "last"),
            has_anomaly=("label_day", "max"),
        )
    )
    candidate_frame = candidate_frame[candidate_frame["active_days"] >= minimum_active_days].copy()
    if len(candidate_frame) < count:
        raise ValueError(
            f"Not enough users with >= {minimum_active_days} active days. "
            f"Found {len(candidate_frame)}. Increase --max-rows-per-file or use a broader sampling mode."
        )
    if strategy == "top-active":
        return candidate_frame.sort_values(["active_days", "user"], ascending=[False, True]).head(count)["user"].tolist()

    rng = np.random.default_rng(seed)
    candidate_frame["random_order"] = rng.random(len(candidate_frame))
    if strategy == "role-stratified-anomaly-demo":
        candidate_frame = candidate_frame.sort_values(["has_anomaly", "random_order"], ascending=[False, True])
    else:
        # Scientific default: holdout selection must not inspect insider labels.
        candidate_frame = candidate_frame.sort_values("random_order")
    selected = []
    # Round-robin by role avoids selecting only the most active role.
    role_groups = {role: group.copy() for role, group in candidate_frame.groupby("role", sort=True)}
    while len(selected) < count:
        progressed = False
        for role in sorted(role_groups):
            group = role_groups[role]
            if group.empty:
                continue
            selected.append(str(group.iloc[0]["user"]))
            role_groups[role] = group.iloc[1:]
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break
    return selected

def main():
    args=parse_args(); out_dir=Path(args.out_dir); holdout_dir=out_dir/'holdout'
    out_dir.mkdir(parents=True, exist_ok=True); holdout_dir.mkdir(parents=True, exist_ok=True)
    configure_feature_rules(args.feature_rules)
    insiders_file = Path(args.insiders_file) if args.insiders_file else Path(args.cert_dir) / "insiders.csv"
    print('[STEP 00] Build unified event log + role context', flush=True)
    events, role_context = build_unified_event_log(
        args.cert_dir,
        args.max_rows_per_file,
        args.ldap_dir,
        sampling_mode=args.sampling_mode,
        random_seed=args.random_seed,
        insiders_file=insiders_file,
    )
    written_outputs=[]
    if args.output_profile == "audit":
        events.to_csv(out_dir/'00_unified_event_log.csv', index=False)
        written_outputs.append('00_unified_event_log.csv')
    role_context.to_csv(out_dir/'role_context.csv', index=False)
    written_outputs.append('role_context.csv')
    print('[STEP 01] Build user-day count/statistical features', flush=True)
    features=build_user_day_features(events)
    if args.output_profile == "audit":
        features.to_csv(out_dir/'01_user_day_features.csv', index=False)
        written_outputs.append('01_user_day_features.csv')
    print('[STEP 02] Build user-day behavior sequences', flush=True)
    seq=build_user_day_sequences(events)
    if args.output_profile == "audit":
        seq.to_json(out_dir/'02_user_day_sequences.jsonl', orient='records', lines=True, force_ascii=False)
        written_outputs.append('02_user_day_sequences.jsonl')
    print('[STEP 03] Join count + sequence + role context', flush=True)
    mv=build_user_day_multiview(features, seq)
    mv = attach_cert_labels(mv, insiders_file, dataset_version="4.2")
    holdout_users = choose_holdout_users(
        mv,
        args.holdout_users,
        args.min_active_days,
        args.holdout_strategy,
        args.random_seed,
    )
    global_mv=mv[~mv['user'].isin(holdout_users)].copy()
    global_mv.to_csv(out_dir/'03_user_day_multiview.csv', index=False)
    written_outputs.append('03_user_day_multiview.csv')
    if args.output_profile == "audit":
        # Optional debug alias; compact mode avoids duplicating the largest table.
        events[~events['user'].isin(holdout_users)].to_csv(out_dir/'global_events.csv', index=False)
        written_outputs.append('global_events.csv')
    users_index=[]
    for user in holdout_users:
        user_events=events[events['user']==user].copy()
        user_file=holdout_dir/f'{sanitize(user)}.csv'
        user_events.to_csv(user_file, index=False)
        user_mv=mv[mv['user']==user]
        users_index.append({
            'userId': user,
            'activeDays': int(user_mv['date'].nunique()),
            'logCount': int(len(user_events)),
            'role': str(user_mv['role'].dropna().iloc[-1]) if 'role' in user_mv and len(user_mv)>0 else 'UNKNOWN',
            'department': str(user_mv['department'].dropna().iloc[-1]) if 'department' in user_mv and len(user_mv)>0 else 'UNKNOWN',
            'file': str(user_file)
        })
        print(f'[WRITE] holdout user {user}: {len(user_events)} logs / {user_mv["date"].nunique()} days -> {user_file}', flush=True)
    (holdout_dir/'index.json').write_text(json.dumps({'users': users_index}, indent=2), encoding='utf-8')
    summary={
        'mode':'multiview_fast_prepare', 'max_rows_per_file':args.max_rows_per_file,
        'sampling_mode': args.sampling_mode, 'holdout_strategy': args.holdout_strategy,
        'holdout_label_aware': args.holdout_strategy == 'role-stratified-anomaly-demo',
        'output_profile': args.output_profile,
        'random_seed': args.random_seed,
        'total_events_after_sampling': int(len(events)), 'total_users_after_sampling': int(events['user'].nunique()),
        'labeled_anomaly_user_days': int(mv['label_day'].sum()),
        'global_user_day_rows': int(len(global_mv)), 'holdout_users': users_index,
        'min_active_days': args.min_active_days,
        'outputs': written_outputs + ['holdout/index.json']
    }
    (out_dir/'prepare_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('[DONE]', flush=True); print(json.dumps(summary, indent=2))
if __name__=='__main__': main()
