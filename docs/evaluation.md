# Evaluation: hai luồng không được trộn kết quả

Thư mục này thuộc evaluation plane, tách khỏi runtime backend. API, ingestion worker và scorer
không được nhận credential của evaluation database. Label, scenario và answer key chỉ được đọc
sau khi scoring hoàn tất; chúng không được sao chép vào core schema, canonical event, feature,
sequence, reference, threshold hoặc model input.

Hiện có hai công cụ metric với hai schema khác nhau:

1. `evaluate_multiview.py` trong sibling experiment cũ, dùng score file kiểu cũ đã chứa label.
2. `machine_learning/cli/evaluate_metrics.py` trong workspace này, dùng decision của framework mới và label
   positive-only được giữ riêng.

Kết quả của luồng 1 không phải kết quả của framework Person → Role → Global mới. Tại thời điểm
viết tài liệu này, workspace chưa có score CERT thực của framework mới: core database local
không có risk assessment và repository không kèm prediction/metric output mới. Các lệnh ở phần
luồng 2 mô tả contract có thể chạy khi đã có score và label hợp lệ, không phải tuyên bố hiệu năng.

## Luồng 1 — metric của sibling experiment cũ

Từ thư mục gốc workspace hiện tại, chạy:

```powershell
Set-Location '..\ANOMALY-DETECTION-main\ml'

python .\evaluate_multiview.py `
  --scores A2=..\models\global_rerun\all_scores.csv `
  --out-dir ..\models\global_rerun\evaluation_new `
  --quantiles 0.90,0.95,0.99,0.995,0.999 `
  --calibration-label-policy ignore
```

Có thể lặp `--scores NAME=path.csv` để so sánh nhiều experiment cũ. Mỗi score CSV phải có tối
thiểu:

```text
split,score,label_day,user,date
```

`scenario` là cột tùy chọn. Script fit threshold từ validation score theo từng quantile rồi chỉ
báo metric trên test. Nó tạo:

- `evaluation_metrics.csv`;
- `scenario_recall.csv`;
- `evaluation_summary.json`.

Đây là schema kết quả của pipeline multiview cũ: label `label_day` nằm ngay trong score CSV,
incident được suy ra theo cặp user/scenario, và script không cung cấp Top-K/day hoặc user-block
confidence interval của evaluator mới. Không nhập các file kết quả này vào framework mới như
thể chúng được sinh từ `risk_assessments`. Muốn so sánh nghiên cứu phải khóa cùng split, label
contract, prediction universe và threshold policy trước.

## Luồng 2 — evaluator của framework mới trong workspace

Chạy CLI từ thư mục `machine_learning`:

```powershell
Set-Location .\machine_learning
..\.venv\Scripts\python.exe -m cli.evaluate_metrics --help
```

### Contract prediction

Chọn đúng một nguồn:

| Nguồn | Tham số | Contract |
|---|---|---|
| CSV export | `--predictions-csv` | Bắt buộc `user_id,day,risk`; tùy chọn `feature_level,sequence_level,status,threshold,is_alert` |
| Core SQLite | `--core-sqlite` | Bắt buộc thêm `--organization`, `--model-version` và `--config-version`; chỉ đọc assessment khớp tenant/split/release |

Quy tắc CSV:

- `day` là ISO `YYYY-MM-DD`.
- `risk` là số hữu hạn trong `[0,1]`; để trống nghĩa là `NO_SCORE`.
- `feature_level` và `sequence_level` hợp lệ là `PERSON`, `ROLE`, `GLOBAL`, `NO_SCORE`.
- Mỗi `(user_id, day)` chỉ xuất hiện một lần.

Evaluator còn bắt buộc `--universe-csv`: manifest độc lập của toàn bộ user-day đủ điều kiện,
có đúng hai cột `user_id,day`. Prediction và label đều phải thuộc universe này. Nhờ đó cả
positive lẫn negative bị thiếu prediction đều được tính là `NO_SCORE`; evaluator không thể
đánh giá đẹp giả tạo bằng cách chỉ xuất các dòng đã score.

Universe chuẩn được materialize từ LDAP effective-dated, không từ event table:

```powershell
Set-Location .\backend
..\.venv\Scripts\python.exe -m cli.export_universe `
  --split VALIDATION `
  --output ..\data\evaluation\validation_universe.csv
```

Ngày nhân viên có hiệu lực nhưng không có event vẫn xuất hiện trong file và không bị loại
khỏi denominator.

Ví dụ tối thiểu:

```csv
user_id,day,risk,feature_level,sequence_level,status
U001,2010-06-01,0.873,PERSON,ROLE,SCORED
U002,2010-06-01,,GLOBAL,NO_SCORE,NO_SCORE
```

### Contract label positive-only

Chọn đúng một nguồn:

| Nguồn | Tham số | Contract |
|---|---|---|
| CSV tách riêng | `--labels-csv` | Bắt buộc `user_id,day`; tùy chọn `incident_id,scenario` |
| Evaluation SQLite | `--labels-sqlite` | Đọc bảng `evaluation_labels` tạo từ `schema.sql` |

Mỗi dòng label là một **positive user-day**. Dòng `(user_id, day)` thuộc universe nhưng không có
trong answer key được xem là negative. Vì loader không có cột nhãn 0/1, file chứa cả dòng âm
sẽ bị hiểu sai thành positive.

CLI bắt buộc flag sau để người chạy xác nhận rõ contract này:

```text
--positive-only-answer-key
```

Flag không tự kiểm chứng answer key; trách nhiệm của người vận hành là mở rộng answer key đúng
thành các positive user-day. Mỗi khóa chỉ được xuất hiện một lần. Mọi user-day trong universe
không có prediction đều bị phạt như `NO_SCORE` với `risk=0`.

Ví dụ:

```csv
user_id,day,incident_id,scenario
U001,2010-06-01,INC-001,S1
U001,2010-06-02,INC-001,S1
```

`../data/evaluation/schema.sql` định nghĩa database label tách vật lý cùng các bảng run/metric. CLI hiện
chỉ đọc `evaluation_labels` từ SQLite và ghi report JSON; nó chưa tự tạo hoặc ghi
`evaluation_runs` và `evaluation_metrics`, cũng chưa đọc PostgreSQL trực tiếp.

### Chọn threshold trên Validation

Primary rule chọn nearest-rank empirical quantile `0,995` trên phân phối risk Validation đã
score, không đọc label khi chọn threshold:

```powershell
..\.venv\Scripts\python.exe -m cli.evaluate_metrics `
  --predictions-csv ..\data\evaluation\validation_predictions.csv `
  --labels-csv ..\data\evaluation\labels\positive_labels.csv `
  --positive-only-answer-key `
  --universe-csv ..\data\evaluation\validation_universe.csv `
  --split VALIDATION `
  --validation-quantile 0.995 `
  --top-k 10 `
  --bootstrap 1000 `
  --bootstrap-seed 20260728 `
  --output ..\data\artifacts\validation.metrics.json
```

`threshold_selection` ghi `labels_used=false`, sample count, nearest rank, tail rate thực tế và
tie overflow. Có thể sensitivity-test `0,99` và `0,999` trên Validation, nhưng `0,995` là
primary đã khóa. `--alert-budget-per-day` vẫn được giữ như phân tích vận hành phụ, không phải
kết quả primary.

### Khóa threshold rồi đánh giá Test

Lấy chính xác `threshold` đã ghi trong report Validation và truyền nguyên giá trị đó sang Test:

```powershell
$validationReport = Get-Content `
  -LiteralPath '..\data\artifacts\validation.metrics.json' `
  -Encoding UTF8 |
  ConvertFrom-Json
$lockedThreshold = [double]$validationReport.threshold

..\.venv\Scripts\python.exe -m cli.evaluate_metrics `
  --predictions-csv ..\data\evaluation\test_predictions.csv `
  --labels-csv ..\data\evaluation\labels\positive_labels.csv `
  --positive-only-answer-key `
  --universe-csv ..\data\evaluation\test_universe.csv `
  --split TEST `
  --threshold $lockedThreshold `
  --top-k 10 `
  --bootstrap 1000 `
  --bootstrap-seed 20260728 `
  --output ..\data\artifacts\test.metrics.json
```

CLI từ chối cả `--alert-budget-per-day` và `--validation-quantile` trên Test. Không chạy nhiều
threshold trên Test rồi chọn kết quả tốt nhất; threshold, Top-K budget và các policy metric
phải được khóa từ Validation trước khi mở báo cáo Test.

### Biến thể đọc hai SQLite tách biệt

Core và label database vẫn được mở riêng; evaluator không attach hoặc ghi label về core:

```powershell
..\.venv\Scripts\python.exe -m cli.evaluate_metrics `
  --core-sqlite ..\data\runtime\insider_threat.db `
  --organization default `
  --model-version baseline.v1 `
  --config-version framework.v4 `
  --labels-sqlite D:\isolated-evaluation\labels.db `
  --positive-only-answer-key `
  --universe-csv ..\data\evaluation\validation_universe.csv `
  --split VALIDATION `
  --validation-quantile 0.995 `
  --top-k 10 `
  --bootstrap 1000 `
  --output ..\data\artifacts\validation.sqlite.metrics.json
```

## Giả định metric đã khóa trong evaluator mới

| Nhóm | Cách tính hiện tại |
|---|---|
| Evaluation universe | Manifest `--universe-csv` độc lập; prediction/label ngoài universe bị từ chối; user-day vắng label là negative |
| `NO_SCORE`/missing prediction | Effective risk bằng `0` để ranking vẫn phạt coverage, nhưng không bao giờ tạo alert hay được chọn vào Top-K; cả positive và negative thiếu prediction vẫn nằm trong denominator |
| AUPRC | Tie-grouped average precision trên toàn bộ user-day |
| ROC-AUC | User-day ROC-AUC; trả `null` nếu không có đủ cả positive và negative |
| Threshold decision | Chỉ row có score mới alert khi `risk >= threshold`; trên Test, `threshold/is_alert` đã lưu (nếu có) phải khớp threshold khóa |
| Precision/Recall/F1 | Micro confusion matrix trên user-day tại threshold đã khóa |
| False alert rate | `false_positive / toàn bộ user-day × 1.000` |
| Alerts/day | Tổng alert chia số ngày khác nhau |
| Recall@Top-K/day | Xếp hạng riêng từng ngày theo risk giảm dần, tie-break bằng `user_id`; lấy tối đa K mỗi ngày rồi tính **micro recall** trên toàn bộ positive user-day |
| Incident detection | Chỉ label có `incident_id`; incident được phát hiện nếu có ít nhất một labeled positive day đạt threshold |
| Detection delay | Số ngày từ positive labeled day đầu tiên của incident đến alerted labeled day đầu tiên; incident không phát hiện bị loại khỏi mean/median delay nhưng vẫn nằm trong denominator detection rate |
| Scenario recall | Day-level recall trên positive user-day có `scenario` |
| Fallback rate | Tỷ lệ `PERSON/ROLE/GLOBAL/NO_SCORE` báo riêng cho Feature và Sequence trên evaluation universe |
| 95% CI | Seeded user-block bootstrap cho AUPRC, ROC-AUC, Precision, Recall và F1 |

Confidence interval hiện không được tạo cho Top-K, operations, incident, scenario hoặc fallback
metric. Bootstrap lấy mẫu user có hoàn lại và giữ toàn bộ ngày của user trong cùng block; đây là
giả định user-block CI, không phải day-block hay incident-block CI.

Report JSON dùng schema `evaluation.metrics.v1`, gồm coverage/denominator, warnings,
`threshold_selection`, metric contract và `manifest_checksum`. Khi không có positive, AUPRC và
recall không xác định; khi không có negative, ROC-AUC không xác định. Không được thay các giá trị
`null` bằng 0 hoặc bỏ warning khi báo cáo.

## Entrypoint tự động

`python -m cli.run_experiment` thực hiện đúng thứ tự tách label:

1. build store, train và fit reference chỉ từ Train;
2. score Validation trước khi materialize answer key;
3. chọn quantile `0.995` trong evaluator trước khi loader đọc label;
4. score Test bằng model/reference đã khóa;
5. truyền nguyên threshold Validation sang Test.

Entrypoint không cho chạy metric từ store bị row-cap, store thiếu Test end hoặc split/universe
không đầy đủ.
