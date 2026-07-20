from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from multiview_features import build_unified_event_log, build_user_day_features, build_user_day_sequences, build_user_day_multiview

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--cert-dir", required=True)
    p.add_argument("--ldap-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--holdout-users", type=int, default=2)
    p.add_argument("--min-active-days", type=int, default=30)
    p.add_argument("--max-rows-per-file", type=int, default=200_000)
    return p.parse_args()

def sanitize(value: str) -> str:
    invalid='<>:"/\\|?*'
    return ''.join('_' if c in invalid else c for c in str(value))

def main():
    args=parse_args(); out_dir=Path(args.out_dir); holdout_dir=out_dir/'holdout'
    out_dir.mkdir(parents=True, exist_ok=True); holdout_dir.mkdir(parents=True, exist_ok=True)
    print('[STEP 00] Build unified event log + role context', flush=True)
    events, role_context = build_unified_event_log(args.cert_dir, args.max_rows_per_file, args.ldap_dir)
    events.to_csv(out_dir/'00_unified_event_log.csv', index=False)
    role_context.to_csv(out_dir/'role_context.csv', index=False)
    print('[STEP 01] Build user-day count/statistical features', flush=True)
    features=build_user_day_features(events); features.to_csv(out_dir/'01_user_day_features.csv', index=False)
    print('[STEP 02] Build user-day behavior sequences', flush=True)
    seq=build_user_day_sequences(events); seq.to_json(out_dir/'02_user_day_sequences.jsonl', orient='records', lines=True, force_ascii=False)
    print('[STEP 03] Join count + sequence + role context', flush=True)
    mv=build_user_day_multiview(features, seq)
    active_counts=mv.groupby('user')['date'].nunique().sort_values(ascending=False)
    candidates=active_counts[active_counts>=args.min_active_days]
    if len(candidates)<args.holdout_users:
        raise ValueError(f'Not enough users with >= {args.min_active_days} active days. Found {len(candidates)}. Increase --max-rows-per-file.')
    holdout_users=list(candidates.head(args.holdout_users).index)
    global_mv=mv[~mv['user'].isin(holdout_users)].copy()
    global_mv.to_csv(out_dir/'03_user_day_multiview.csv', index=False)
    # Keep global_events alias for old eyes/debug.
    events[~events['user'].isin(holdout_users)].to_csv(out_dir/'global_events.csv', index=False)
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
        'total_events_after_sampling': int(len(events)), 'total_users_after_sampling': int(events['user'].nunique()),
        'global_user_day_rows': int(len(global_mv)), 'holdout_users': users_index,
        'min_active_days': args.min_active_days,
        'outputs': ['00_unified_event_log.csv','01_user_day_features.csv','02_user_day_sequences.jsonl','03_user_day_multiview.csv','role_context.csv']
    }
    (out_dir/'prepare_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('[DONE]', flush=True); print(json.dumps(summary, indent=2))
if __name__=='__main__': main()
