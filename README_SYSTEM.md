# Tài liệu hệ thống Insider Threat

Tài liệu này mô tả trạng thái kiến trúc hiện tại của repository, trách nhiệm của từng
thư mục và luồng dữ liệu từ CERT R4.2 đến kết quả đánh giá. Hướng dẫn cài đặt và chạy
từng bước nằm trong `README_TUTORIAL.md`.

## 1. Mục tiêu của hệ thống

Hệ thống nghiên cứu phát hiện bất thường hành vi nội bộ theo đơn vị `employee-day`.
Mô hình chính là **TCN–Transformer Autoencoder** không giám sát, nhận đồng thời:

- `Feature128`: 128 đặc trưng tổng hợp hành vi của một nhân viên trong một ngày.
- `Sequence7`: chuỗi tối đa 256 event trong ngày, gồm 7 loại event khách quan.
- Cửa sổ thời gian cố định 30 ngày: `[D-29,D]`.

Mô hình không học trực tiếp từ nhãn insider. CERT answer key chỉ được đưa vào sau khi
đã tạo prediction, tại bước evaluation.

Hai reconstruction error được giữ riêng:

- Feature reconstruction error.
- Sequence reconstruction error.

Mỗi nhánh tự chọn reference theo thứ tự:

```text
Person → Role → Global → NO_SCORE
```

Sau khi hai nhánh được calibrate độc lập, các calibrated percentile mới được fusion
thành `risk`.

## 2. Luồng framework tổng thể

```text
CERT R4.2 raw
  ├─ logon.csv
  ├─ device.csv
  ├─ file.csv
  ├─ http.csv
  ├─ email.csv
  ├─ LDAP/*.csv
  └─ answers/*                 chỉ dành cho evaluation
          │
          ▼
Disk-backed streaming materialization
  ├─ aggregate từng source/user/day
  ├─ Train-only PC profiles
  ├─ past-only temporal/history state
  └─ daily Feature128 + Sequence7 tensors
          │
          ▼
SQLite user-day store
          │
          ▼
SQLiteWindowDataset
  └─ tạo cửa sổ [D-29,D] khi được yêu cầu
          │
          ▼
TCN–Transformer Autoencoder
  ├─ Feature raw reconstruction score
  └─ Sequence raw reconstruction score
          │
          ▼
Frozen Train references
  ├─ Person reference theo role epoch
  ├─ Role reference
  └─ Global reference
          │
          ▼
Validation calibrated predictions
  └─ chọn threshold empirical quantile 0.995, không dùng label
          │
          ▼
Test calibrated predictions
  └─ dùng nguyên threshold đã khóa từ Validation
          │
          ▼
Research metrics + experiment report
```

## 3. Cấu trúc repository

```text
Insider threat/
├─ README.md
├─ README_SYSTEM.md
├─ README_TUTORIAL.md
├─ compose.yaml
├─ backend/
├─ machine_learning/
├─ data/
└─ docs/
```

### 3.1. Thư mục `machine_learning/`

Chứa toàn bộ code chuẩn bị tensor, mô hình, train, inference và evaluation nghiên cứu.
Package này không phụ thuộc FastAPI hoặc ORM của backend.

```text
machine_learning/
├─ config/
│  ├─ feature128.v5.json
│  ├─ sequence7.v4.json
│  └─ model.tcn_transformer_ae.v4.json
├─ insider_ml/
│  ├─ contracts.py
│  ├─ cert_data.py
│  ├─ preprocessing.py
│  ├─ temporal.py
│  ├─ stream_store.py
│  ├─ dataset.py
│  ├─ model.py
│  ├─ training.py
│  ├─ inference.py
│  └─ evaluation/
│     └─ metrics.py
├─ cli/
│  ├─ build_store.py
│  ├─ prepare_cert.py
│  ├─ train.py
│  ├─ score.py
│  ├─ prepare_evaluation.py
│  ├─ evaluate_metrics.py
│  ├─ run_framework.py
│  └─ run_experiment.py
├─ tests/
└─ pyproject.toml
```

#### Các file cấu hình

| File | Trách nhiệm |
|---|---|
| `feature128.v5.json` | Khóa tên, thứ tự và chính sách 128 feature |
| `sequence7.v4.json` | Khóa vocabulary, side channel, event order và max length |
| `model.tcn_transformer_ae.v4.json` | Khóa kiến trúc model, kích thước embedding và loss |

#### Các module lõi

| File | Trách nhiệm |
|---|---|
| `contracts.py` | Shape, dtype, schema version, split và cửa sổ 30 ngày |
| `cert_data.py` | Materializer NPZ legacy; chỉ phù hợp smoke hoặc shard nhỏ |
| `preprocessing.py` | PC context, metadata feature, robust scaler và helper |
| `temporal.py` | Biểu diễn thời gian tuần hoàn và temporal anomaly baseline |
| `stream_store.py` | Materializer chính ít RAM, ghi aggregate/tensor vào SQLite |
| `dataset.py` | Đọc NPZ hoặc dựng cửa sổ trực tiếp từ SQLite |
| `model.py` | TCN–Transformer Autoencoder |
| `training.py` | Reconstruction loss và một epoch train |
| `inference.py` | Raw reconstruction score và empirical-CDF calibration |
| `evaluation/metrics.py` | Metric user-day, incident, coverage và bootstrap |

#### Các CLI

| Lệnh | Chức năng |
|---|---|
| `cli.build_store` | Đọc toàn bộ raw CERT theo streaming và tạo SQLite store |
| `cli.prepare_cert` | Nhánh NPZ legacy; không dùng cho toàn bộ HTTP 15 GB |
| `cli.train` | Train/resume model từ SQLite store hoặc NPZ |
| `cli.score` | Sinh raw score, fit frozen Train reference hoặc calibrate |
| `cli.prepare_evaluation` | Tạo LDAP universe và positive-only answer-key labels |
| `cli.evaluate_metrics` | Chọn/khóa threshold và tính metric |
| `cli.run_framework` | Smoke end-to-end có giới hạn trên dữ liệu CERT thật |
| `cli.run_experiment` | Chạy toàn bộ pipeline đến Test metric |

`cli.prepare_cert` mặc định giới hạn 25 user và luôn ghi `max_users` vào manifest. Artifact từ
user shard chỉ dùng cho smoke/research; production reference yêu cầu nguồn không giới hạn user
và pipeline SQLite đầy đủ.

### 3.2. Thư mục `backend/`

Chứa lớp vận hành: FastAPI, persistence, alert workflow, audit và safe-update
production. Backend không train model.

```text
backend/
├─ app/
│  ├─ main.py
│  ├─ api.py
│  ├─ schemas.py
│  ├─ models.py
│  ├─ services.py
│  ├─ validation.py
│  ├─ database.py
│  ├─ settings.py
│  ├─ catalog.py
│  └─ domain/
│     ├─ readiness.py
│     ├─ fusion.py
│     └─ universe.py
├─ cli/
│  ├─ init_db.py
│  ├─ ingest_jsonl.py
│  └─ export_universe.py
├─ config/
│  ├─ framework.v4.json
│  ├─ framework.v5.json
│  └─ framework.v6.json
├─ migrations/
├─ tests/
├─ .env.example
├─ alembic.ini
├─ Dockerfile
└─ pyproject.toml
```

| Thành phần | Trách nhiệm |
|---|---|
| `api.py` | REST endpoint cho ingestion, feature, sequence, reference, score và alert |
| `models.py` | ORM schema cho database |
| `schemas.py` | Request/response validation |
| `services.py` | Nghiệp vụ, persistence, audit và transaction |
| `validation.py` | Label firewall, metadata-only và tensor contract |
| `domain/readiness.py` | Chọn Person → Role → Global độc lập cho từng nhánh |
| `domain/fusion.py` | Fusion calibrated Feature/Sequence score |
| `domain/universe.py` | Universe employee-day theo role assignment có hiệu lực |
| `framework.v5.json` | Contract baseline của `experiment_v1`; giữ nguyên để tái lập kết quả cũ |
| `framework.v6.json` | Contract nghiên cứu chính: readiness theo 124 feature đang bật và inactive day là `NO_SCORE` |

Backend có thể lưu reference, nhận raw branch error, calibrate, fusion, áp dụng
threshold đã khóa và tạo alert. `run_experiment` hiện xuất CSV/JSON phục vụ nghiên cứu;
nó chưa tự động import kết quả vào backend database.

`framework.v6` hiện là contract cho offline research pipeline và đặt
`safe_personalized_update.production_enabled=false`. Backend runtime vẫn mặc định v5 cho đến
khi có migration reference/database và rollout production riêng; không dùng v6 research config
để bật production trực tiếp.

### 3.3. Thư mục `data/`

Không đặt source code trong `data/`. Những file lớn và artifact sinh ra đã được
`.gitignore`.

```text
data/
├─ raw/
│  └─ cert4.2/                    junction đến dataset thật
├─ interim/                       dữ liệu trung gian nếu có
├─ processed/
│  ├─ cert4.2_user_days.sqlite    full disk-backed store
│  └─ smoke/                      store smoke
├─ artifacts/
│  ├─ experiment_v1/              baseline framework.v5, không ghi đè
│  ├─ experiment_v2/              primary framework.v6
│  └─ smoke/                      artifact smoke
├─ evaluation/
│  ├─ schema.sql
│  ├─ experiment_v1/              universe và label baseline
│  └─ experiment_v2/              universe và label primary v6
└─ runtime/
   └─ insider_threat.db           database backend local
```

#### `data/raw`

Chỉ đọc. Dataset không được sửa hoặc ghi đè. Trên Windows nên dùng directory
junction để dataset chỉ tồn tại một bản.

#### `data/processed`

Chứa dữ liệu model-ready. Pipeline full hiện dùng SQLite thay vì một NPZ khổng lồ.

#### `data/artifacts`

Chứa:

- Robust scaler và checksum.
- Model checkpoint `.pt`.
- Model manifest.
- Frozen Train references.
- Raw/calibrated prediction CSV.
- Validation/Test metric JSON.
- `experiment_report.json`.

#### `data/evaluation`

Chứa universe và answer-key label phục vụ evaluation. Label không được đưa vào
preprocessing, model input, train, reference hoặc threshold selection.

#### `data/runtime`

Chứa database và log vận hành của backend, không phải input train.

### 3.4. Thư mục `docs/`

```text
docs/
├─ architecture.md
├─ evaluation.md
├─ Framework_sua_doi_Insider_Threat_Person_Role_Global.docx
└─ tools/
```

- `architecture.md`: kiến trúc và rule chi tiết.
- `evaluation.md`: label firewall, universe, threshold và metric.
- File DOCX: tài liệu nghiên cứu/BA.
- `tools/`: script tạo tài liệu, không nằm trong runtime pipeline.

## 4. Hợp đồng Feature128

Mỗi employee-day có vector 128 chiều:

| Nhóm | Số chiều | ID |
|---|---:|---|
| Logon | 18 | L01–L18 |
| Device | 12 | D01–D12 |
| File | 24 | F01–F24 |
| HTTP | 24 | H01–H24 |
| Email | 26 | E01–E26 |
| Cross-source | 16 | X01–X16 |
| History | 8 | P01–P08 |
| Tổng | 128 | |

Các nguyên tắc:

- Count/sum có thể là số 0 đã quan sát.
- Statistic hoặc ratio không xác định được biểu diễn bằng `value=0` và
  `feature_mask=false`.
- Scaler là median/MAD, chỉ fit từ Train.
- History chỉ dùng ngày trước D.
- Không dùng raw content trong primary experiment.
- OCEAN chỉ dành cho ablation riêng.
- Extension category bị tắt khi chưa có mapping versioned.
- X12–X15 giữ vị trí schema nhưng bị mask trong primary experiment.
- Không có rule cố định `after_hours`; dùng độ lệch tương đối với baseline quá khứ.

## 5. Hợp đồng Sequence7

Bảy event token:

```text
LOGON
LOGOFF
DEVICE_CONNECT
DEVICE_DISCONNECT
FILE
HTTP
EMAIL
```

Side channel của từng event:

- PC context: `OWN`, `SHARED`, `FOREIGN`, `UNKNOWN`.
- Calendar: `WEEKDAY`, `WEEKEND`.
- Gap bucket: `0-1`, `1-5`, `5-30`, `30-120`, `>120` phút.
- `time_sin` và `time_cos` từ local wall-clock CERT.

Mỗi ngày có tối đa 256 event. Nếu vượt quá giới hạn, giữ 128 event đầu và 128
event cuối. Thứ tự chuẩn:

```text
timestamp → source_rank → event_uid
```

HTTP không bị gán nhãn `RISK`. Novelty domain được tính ở Feature branch bằng lịch sử
past-only.

## 6. PC context Train-only

Mỗi PC được thống kê bằng distinct user-day trong Train:

```text
dominance = user-days của user dùng nhiều nhất / tổng user-days của PC
```

Rule:

- `SHARED`: distinct users ≥ 5 và dominance < 0.5.
- `OWN`: user là frozen owner và owner dominance ≥ 0.5.
- `FOREIGN`: PC có owner khác user hiện tại.
- `UNKNOWN`: không đủ dữ liệu kết luận.

PC profile được freeze sau Train và không fit lại trên Validation/Test.

## 7. Universe, split và cửa sổ

Đơn vị đánh giá là mọi `(employee, day)` mà employee có hiệu lực trong LDAP.
Ngày không có event vẫn là inactive employee-day và không bị loại khỏi denominator.

Split cố định, inclusive:

| Split | Bắt đầu | Kết thúc |
|---|---|---|
| Train | 2010-01-02 | 2010-05-31 |
| Validation | 2010-06-01 | 2010-09-30 |
| Test | 2010-10-01 | 2011-05-17 |

Mỗi sample là cửa sổ đúng 30 ngày `[D-29,D]`. Không sử dụng ngày sau D.

## 8. Kiến trúc model

### Feature encoder

```text
128 values + 128 masks
→ Linear(256,128)
→ GELU + LayerNorm
→ Linear(128,64)
```

### Sequence encoder

Mỗi event có embedding:

```text
event token 24
+ PC context 4
+ calendar 3
+ gap 4
+ cyclic time 2
= 37 chiều
```

Event TCN/Conv1D mã hóa chuỗi trong ngày, sau đó masked mean pooling tạo embedding
64 chiều/ngày.

### Temporal encoder

```text
Feature embedding 64 + Sequence embedding 64
→ day embedding 128
→ residual TCN theo 30 ngày, dilation [1,2,4,8]
→ Transformer 2 layers, 4 heads, FFN 256
```

### Decoder và loss

Decoder tái tạo:

- Feature128.
- Event token.
- PC context.
- Calendar context.
- Gap bucket.
- Cyclic time.

```text
total_loss = 0.5 × feature_loss + 0.5 × sequence_loss
```

Các component Sequence có trọng số bằng nhau. PAD không tham gia loss.

## 9. Train, score và ba tầng reference

Train không dùng label. Với store full, endpoint Train được lấy theo lịch tuần,
stagger theo hash user để giảm số sample; mỗi sample vẫn reconstruct đủ 30 ngày.

Inference chỉ lấy reconstruction error của ngày cuối D:

```text
feature_raw(D)
sequence_raw(D)
```

Ba tầng là ba phân phối calibration, không phải ba model:

- `Person`: cùng user và role epoch.
- `Role`: những user có cùng role.
- `Global`: toàn bộ Train.

Feature và Sequence tự kiểm tra readiness, vì vậy một ngày có thể dùng:

```text
Feature = PERSON
Sequence = ROLE
```

hoặc bất kỳ tổ hợp hợp lệ nào khác.

References của primary Validation/Test đều fit Train-only và frozen.

## 10. Fusion, threshold và evaluation

Raw error được chuyển thành empirical CDF percentile trong selected reference.
Giá trị càng gần 1 nghĩa là error càng cao so với Train reference.

Khi đủ cả hai nhánh:

```text
risk = 0.5 × q_feature + 0.5 × q_sequence
```

Nếu chỉ một nhánh có score, trọng số được chuẩn hóa thành 1. Nếu cả hai không có
score thì kết quả là `NO_SCORE`.

Validation chọn threshold bằng empirical quantile `0.995` mà không đọc label. Test
chỉ sử dụng threshold đã khóa.

Positive label là incident subject có ít nhất một answer-key event trong ngày D.
Không mở rộng toàn bộ incident interval và không coi actor phụ là positive.

Metric gồm:

- AUPRC và ROC-AUC.
- Precision, Recall, F1, TP, FP, TN, FN.
- Recall@Top-K/day.
- Alerts/day và false alerts/1.000 user-days.
- Incident detection rate và time-to-detect.
- Scenario recall.
- Person/Role/Global/NO_SCORE rate.
- User-block bootstrap confidence interval.

Missing prediction được tính là `NO_SCORE` với effective risk bằng 0 để coverage
failure không làm metric đẹp lên giả tạo.

## 11. Backend và research pipeline

Research pipeline:

```text
raw CERT → SQLite store → checkpoint → CSV/JSON prediction → metric report
```

Operational backend:

```text
canonical event/reference/raw branch score
→ server-side readiness/calibration/fusion
→ locked threshold
→ assessment/alert/audit
```

Hai phía dùng chung schema/rule version nhưng hiện không có job tự động import artifact
của `run_experiment` vào backend.

Safe personalized update bị loại khỏi primary experiment. Backend chỉ cho phép workflow
này ở production, sau cửa sổ quarantine đóng `[D,D+30]`; ngày sớm nhất đủ điều
kiện là `D+31`, với đầy đủ scoring watermark và không có alert trong cửa sổ.

## 12. Artifact, version và khả năng resume

Store ghi checkpoint hoàn thành theo từng raw source. Model checkpoint được ghi nguyên
tử sau mỗi epoch. Score CSV có manifest checksum.

Khi chạy lại:

- Source raw đã hoàn thành và fingerprint không đổi sẽ được bỏ qua.
- Epoch đã checkpoint sẽ được resume.
- Prediction có checkpoint/reference/output checksum hợp lệ sẽ được tái sử dụng.
- Checksum không khớp sẽ bị từ chối thay vì trộn artifact cũ và mới.

## 13. Trạng thái hiện tại

Đã có:

- Code full pipeline đến Test metric.
- Smoke store, checkpoint, reference và prediction.
- Smoke end-to-end trên dữ liệu CERT thật: `PASS`.
- Backend database và API implementation.
- Unit/integration tests cho ML và backend.

Chưa có:

- Full `data/processed/cert4.2_user_days.sqlite`.
- Primary `data/artifacts/experiment_v2/experiment_report.json` với framework.v6.
- Metric nghiên cứu thật trên toàn bộ CERT R4.2.
- Job deployment tự động nối research artifact vào backend.

Smoke chỉ xác nhận code path chạy được, không phải bằng chứng hiệu năng phát hiện.
