# ML pipeline – CERT R4.2 multi-view anomaly detection

Pipeline chính gồm ba bước: prepare dữ liệu, train global model và đánh giá bằng
nhãn CERT. Các script không có hậu tố `multiview` là phiên bản legacy.

## 1. Prepare dữ liệu

Chế độ mặc định `compact` chỉ ghi những artifact cần cho train/runtime, tránh
nhân đôi nhiều GB CSV trung gian:

```powershell
python .\multiview_prepare_fast.py --cert-dir ..\data\cert4.2 --ldap-dir ..\data\ldap --out-dir ..\data\processed --sampling-mode time-user-stratified --max-rows-per-file 1000000 --holdout-strategy role-stratified --output-profile compact --random-seed 42 --insiders-file ..\data\cert4.2\insiders.csv --feature-rules .\feature_rules.json
```

Output compact:

```text
data/processed/03_user_day_multiview.csv
data/processed/role_context.csv
data/processed/prepare_summary.json
data/processed/holdout/index.json
data/processed/holdout/<USER>.csv
```

Nếu cần audit/debug đầy đủ, dùng `--output-profile audit`. Chế độ này ghi thêm
`00_unified_event_log.csv`, `01_user_day_features.csv`,
`02_user_day_sequences.jsonl` và `global_events.csv`, nên cần nhiều dung lượng.

`role-stratified` là holdout khoa học mặc định và không nhìn `label_day`.
`role-stratified-anomaly-demo` chỉ dùng cho demo có chủ đích cần giữ insider
trong holdout; kết quả từ chế độ này phải được ghi rõ là label-aware.

## 2. Train global model bằng lazy windows

Training lưu mỗi calendar day đúng một lần bằng kiểu dữ liệu gọn. Cửa sổ 30
ngày chỉ được dựng khi DataLoader yêu cầu batch, vì vậy batch size giờ thực sự
khống chế RAM và có thể giữ `max-events-per-day=256` trên máy yếu.

```powershell
python .\train_multiview.py --scope global --multiview-csv ..\data\processed\03_user_day_multiview.csv --role-context ..\data\processed\role_context.csv --feature-rules .\feature_rules.json --model-dir ..\models\global --window-size 30 --epochs 3 --batch-size 4 --lr 0.0005 --hidden-dim 128 --transformer-layers 2 --heads 4 --max-events-per-day 256 --view-mode count-sequence --training-label-policy ignore --calibration-label-policy ignore --anomaly-quantile 0.995 --min-role-users 30 --min-role-user-days 1000 --num-workers 0 --progress-every 250 --random-seed 42
```

Hai policy mặc định giữ thí nghiệm unsupervised:

```text
training-label-policy=ignore
calibration-label-policy=ignore
```

Để chạy ablation one-class có làm sạch bằng nhãn lịch sử:

```text
--training-label-policy exclude-positive-windows
--calibration-label-policy benign-only
```

Chế độ clean dùng nhãn để loại positive windows khỏi train/calibration nên
không được gọi là fully unsupervised. Hãy lưu vào model directory khác để so
sánh công bằng với contaminated baseline.

Output:

```text
models/global/model.pt
models/global/vectorizer.joblib
models/global/metadata.json
models/global/all_scores.csv
```

Terminal hiển thị `[DATA]`, `[TRAIN]` và `[SCORE]`. Metadata ghi seed, policy,
loss từng epoch, tỷ lệ ngày bị truncate, dung lượng lazy storage và model config.

## 3. Evaluation

`insiders.csv` đã được prepare chuyển thành `label_day` và `scenario`. Nhãn
không được truyền lại vào evaluation:

```powershell
python .\evaluate_multiview.py --scores A2=..\models\global\all_scores.csv --out-dir ..\models\global\evaluation --quantiles 0.90,0.95,0.99,0.995,0.999
```

`evaluation_metrics.csv` gồm:

```text
precision, recall, f1, auprc
true_positives, false_positives, true_negatives, false_negatives
false_positive_rate, specificity
alerts_per_1000_user_days
false_alerts_per_1000_benign_user_days
total_incidents, detected_incidents, incident_recall
mean_detection_delay_days, median_detection_delay_days
prevalence, auprc_random_baseline, auprc_lift_over_random
```

`scenario_recall.csv` báo cả day recall và incident recall theo scenario.
Threshold luôn được fit từ validation score; test chỉ dùng để báo cáo.

## 4. Personalized calibration và prediction

Personalized vẫn dùng global neural model, chỉ fit robust threshold từ delayed
safe history:

```powershell
python .\train_multiview.py --scope personalized --db ..\data\anomaly.db --user-id USER_ID --model-dir ..\models\personalized\USER_ID --global-model-dir ..\models\global --role-context ..\data\processed\role_context.csv --feature-rules .\feature_rules.json --min-safe-days 30 --safe-history-days 90 --update-delay-days 7
```

```powershell
python .\predict_multiview.py --db ..\data\anomaly.db --user-id USER_ID --scope personalized --model-dir ..\models\personalized\USER_ID --global-model-dir ..\models\global --role-context ..\data\processed\role_context.csv --feature-rules .\feature_rules.json
```

## 5. Nguyên tắc nghiên cứu

- Không chọn threshold cuối cùng dựa trên test metrics.
- Báo sampling mode, row budget, seed, output profile và label policy.
- So sánh ít nhất count-only, sequence-only và count-sequence.
- Báo incident recall và false alerts/1.000 benign user-days, không chỉ accuracy.
- Sampling label-aware chỉ dành cho demo, không dùng để tuyên bố hiệu năng tổng quát.
- `sampling-mode full` vẫn cần RAM lớn ở bước đọc raw CSV; lazy windows giải
  quyết RAM khi train, không biến raw-data preparation thành streaming.
