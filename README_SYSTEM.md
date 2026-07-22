# ANOMALY-DETECTION — CERT R4.2 Multi-view Insider Threat Framework

Dự án này là framework demo phát hiện hành vi bất thường/insider threat trên bộ dữ liệu CERT R4.2. Hệ thống kết hợp ba phần:

1. **Backend ASP.NET Core Web API**: nhận log, lưu SQLite, gọi Python ML worker, expose API/Swagger.
2. **Python ML pipeline**: chuẩn hóa dữ liệu, tạo count view + raw sequence view, train global model, fit/update safe personalized threshold, predict anomaly.
3. **Frontend React/Vite**: giao diện demo trực quan flow Global → Role/Department calibration → Personalized baseline.

Framework hiện tại dùng đơn vị phân tích chính là **user-day**: một user trong một ngày. Mỗi user-day được biểu diễn bằng 2 nhánh:

- **Count/statistical view**: đếm số lượng/tần suất/cường độ hành vi trong ngày.
- **Raw sequence view**: token/source/time-gap đi qua neural daily sequence encoder để giữ thông tin thứ tự.

Hai nhánh được nén thành vector 128 chiều:

```text
64 chiều count + 64 chiều sequence = 128 chiều / user-day
```

Sau đó model tạo window 30 ngày:

```text
X_window = 30 calendar days liên tục × 128 chiều
```

Model chính là **TCN-Transformer Autoencoder**. Model học cách reconstruct hành vi bình thường; hành vi nào reconstruct sai nhiều thì reconstruction error cao và được xem là bất thường.

---

## 1. Kiến trúc tổng quan

```text
5 raw logs CERT R4.2
(logon, device, file, email, http)
        +
LDAP monthly files
(role, department, team, ...)
        ↓
ml/multiview_prepare_fast.py
        ↓
data/processed/00_unified_event_log.csv
        ↓
data/processed/01_user_day_features.csv      ← Count/statistical view
        ↓
data/processed/02_user_day_sequences.jsonl   ← Raw token/source/time-gap sequence view
        ↓
data/processed/03_user_day_multiview.csv     ← Input chính cho global training
        ↓
ml/train_multiview.py
        ↓
models/global/model.pt
models/global/vectorizer.joblib
models/global/metadata.json
        ↓
Backend API + Frontend demo
        ↓
User mới: Global + Role/Department calibration
User đủ lịch sử an toàn: Safe personalized threshold (không train neural model riêng)
```

---

## 2. Cấu trúc thư mục

```text
ANOMALY-DETECTION/
├── backend/
│   └── AnomalyFramework.Api/
│       ├── Data/
│       ├── Dtos/
│       ├── Models/
│       ├── Services/
│       ├── Program.cs
│       ├── appsettings.json
│       └── AnomalyFramework.Api.csproj
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── styles.css
│   ├── package.json
│   └── index.html
│
├── ml/
│   ├── requirements.txt
│   ├── model.py
│   ├── multiview_features.py
│   ├── multiview_prepare_fast.py
│   ├── train_multiview.py
│   ├── predict_multiview.py
│   ├── cert_prepare.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── data/
│   ├── cert4.2/
│   ├── ldap/
│   └── processed/
│
├── models/
│   ├── global/
│   └── personalized/
│
├── README.md
├── README_MULTIVIEW.md
└── .gitignore
```

---

## 3. Các folder chính dùng để làm gì?

| Folder | Ý nghĩa |
|---|---|
| `backend/AnomalyFramework.Api` | Backend ASP.NET Core Web API. Chứa API, SQLite DB context, service gọi Python, logic routing global/personalized. |
| `frontend` | Giao diện demo React/Vite. Dùng để chọn holdout user, replay log theo ngày, xem score/route/anomaly. |
| `ml` | Toàn bộ code Python cho xử lý dữ liệu, train model và predict anomaly. |
| `data/cert4.2` | Nơi đặt 5 raw log CERT: `logon.csv`, `device.csv`, `file.csv`, `email.csv`, `http.csv`. Không push folder này lên GitHub. |
| `data/ldap` | Nơi đặt các file LDAP theo tháng, ví dụ `2010-01.csv`, `2011-05.csv`. Không push data thật nếu nhạy cảm. |
| `data/processed` | Output sau khi prepare dữ liệu: unified event log, user-day features, sequence, multiview, holdout users. Không push generated files. |
| `models/global` | Model global baseline đã train: `model.pt`, `vectorizer.joblib`, `metadata.json`, `all_scores.csv`. Không push model binary lên GitHub nếu nặng. |
| `models/personalized` | Metadata và score history của safe personalized threshold theo user; không chứa neural model riêng. Không push generated artifacts lên GitHub. |

---

## 4. Backend — mô tả từng file

### `backend/AnomalyFramework.Api/Program.cs`

Đây là file khởi động backend ASP.NET Core.

Nhiệm vụ chính:

- Cấu hình SQLite database.
- Đăng ký các service như `TrainingService`, `PredictionService`, `DetectionService`, `DemoReplayService`.
- Bật CORS cho frontend ở `localhost:5173`.
- Bật Swagger.
- Khai báo toàn bộ API endpoint.

Các API chính:

| API | Ý nghĩa |
|---|---|
| `POST /api/logs` | Nhập 1 log thủ công, lưu DB, predict anomaly, có thể trigger personalized training. |
| `POST /api/logs/bulk` | Nhập nhanh nhiều log demo. |
| `GET /api/users/{userId}/status` | Xem trạng thái user: log count, active days, personalized ready, role/department. |
| `GET /api/detections` | Xem lịch sử các lần detect/predict. |
| `GET /api/dashboard/summary` | Lấy thống kê tổng quan cho dashboard frontend. |
| `POST /api/training/global` | Train Global Role/Department baseline. |
| `POST /api/training/personalized/{userId}` | Kiểm tra/fit safe personalized threshold thủ công cho một user. |
| `GET /api/demo/holdout-users` | Lấy danh sách user holdout dùng cho demo. |
| `POST /api/demo/replay` | Replay log của holdout user vào DB theo ngày/event. |

---

### `backend/AnomalyFramework.Api/appsettings.json`

File cấu hình chính cho backend và ML worker.

Các field quan trọng:

```json
{
  "ConnectionStrings": {
    "Default": "Data Source=../../data/anomaly.db"
  },
  "Ml": {
    "PythonExecutable": "python",
    "WorkingDirectory": "../../ml",
    "ModelDirectory": "../../models",
    "GlobalTrainingMultiviewPath": "../../data/processed/03_user_day_multiview.csv",
    "RoleContextPath": "../../data/processed/role_context.csv",
    "HoldoutDirectory": "../../data/processed/holdout",
    "FeatureRulesPath": "../../ml/feature_rules.json",
    "PersonalizedActiveDaysThreshold": 30,
    "WindowSize": 30,
    "TrainEpochs": 3,
    "TrainBatchSize": 32,
    "LearningRate": 0.0005,
    "HiddenDimension": 128,
    "TransformerLayers": 2,
    "AttentionHeads": 4,
    "MaxEventsPerDay": 256,
    "AnomalyQuantile": 0.995,
    "MinRoleUsers": 30,
    "MinRoleUserDays": 1000,
    "PersonalizedSafeHistoryDays": 90,
    "PersonalizedUpdateDelayDays": 7,
    "PersonalizedUpdateEverySafeDays": 7,
    "PredictionTimeoutSeconds": 120,
    "TrainingTimeoutSeconds": 900
  }
}
```

Giải thích nhanh:

| Config | Ý nghĩa |
|---|---|
| `PythonExecutable` | Python mà backend dùng để gọi ML scripts. Local nên trỏ vào `.venv/Scripts/python.exe` nếu bị thiếu package. |
| `WorkingDirectory` | Folder chạy Python scripts, hiện là `ml`. |
| `ModelDirectory` | Folder chứa model global/personalized. |
| `GlobalTrainingMultiviewPath` | File input chính để train global. |
| `RoleContextPath` | File role context được sinh ra từ LDAP. |
| `PersonalizedActiveDaysThreshold` | Pre-check rẻ trước khi worker đếm ngày an toàn; không tự động đồng nghĩa personal threshold đã đủ điều kiện. |
| `WindowSize` | Số calendar days liên tục trong một window model, hiện là 30; ngày trống được zero-fill. |
| `TrainEpochs` | Số vòng học qua toàn bộ dữ liệu. Demo có thể để 1 để tránh timeout. |
| `AnomalyQuantile` | Percentile dùng để lấy anomaly threshold, hiện là 99.5%. |
| `MinRoleUsers`, `MinRoleUserDays` | Role/department chỉ có threshold riêng khi đủ ít nhất 30 user hoặc 1.000 validation user-days. |
| `PersonalizedSafeHistoryDays`, `PersonalizedUpdateDelayDays` | Cửa sổ lịch sử an toàn và độ trễ chống baseline poisoning. |
| `PredictionTimeoutSeconds` | Timeout khi predict. |
| `TrainingTimeoutSeconds` | Timeout khi train model từ API. |

---

### `backend/AnomalyFramework.Api/AnomalyFramework.Api.csproj`

File project .NET. Khai báo target framework và NuGet packages.

Các package chính:

- ASP.NET Core Web API
- Entity Framework Core
- SQLite provider
- Swagger/OpenAPI

---

### `backend/AnomalyFramework.Api/Data/AppDbContext.cs`

DbContext của Entity Framework Core.

Nhiệm vụ:

- Kết nối SQLite DB `data/anomaly.db`.
- Khai báo các bảng:
  - `RawLogs`
  - `UserModelStates`
  - `DetectionResults`

Backend tự tạo DB khi chạy lần đầu bằng `EnsureCreated()` trong `Program.cs`.

---

## 5. Backend Models — các bảng trong DB

### `Models/RawLog.cs`

Đại diện cho log raw/runtime sau khi được insert vào hệ thống.

Một record tương ứng một event được user nhập/replay.

Các field chính:

| Field | Ý nghĩa |
|---|---|
| `Id` | ID nội bộ của raw log. |
| `SourceEventId` | ID event gốc nếu có. |
| `UserId` | User phát sinh hành vi. |
| `PcId` | Máy tính liên quan. |
| `TimestampUtc` | Thời điểm event. |
| `EventType` | Loại log: logon/device/file/email/http hoặc dạng tương tự. |
| `Activity` | Hành động chi tiết, ví dụ Logon, Logoff, Connect. |
| `Url` | URL nếu là http event. |
| `FileName` | Tên file nếu là file event. |
| `EmailTo/Cc/Bcc/From` | Thông tin email nếu là email event. |
| `Size` | Size nếu có. |
| `AttachmentCount` | Số attachment nếu có. |
| `Content` | Nội dung/payload bổ sung. |
| `InputMode` | Nguồn input: manual, replay, bulk. |

---

### `Models/UserModelState.cs`

Lưu trạng thái model của từng user.

Dùng để biết user đang ở global hay personalized.

Các field chính:

| Field | Ý nghĩa |
|---|---|
| `UserId` | User ID. |
| `LogCount` | Tổng số log đã nhận của user. |
| `ActiveDaysCount` | Số ngày user có hoạt động. |
| `Role` | Role hiện tại/role được detect từ LDAP context. |
| `Department` | Department hiện tại. |
| `IsPersonalizedReady` | User đã có safe personal threshold active hay chưa. |
| `IsTraining` | Worker có đang kiểm tra/fit personal threshold hay không. |
| `PersonalizedModelPath` | Tên cũ của field; hiện trỏ tới thư mục calibration metadata, không phải model riêng. |
| `LastPersonalizedTrainingAtUtc` | Thời điểm personal threshold được update gần nhất. |
| `LastTrainingMessage` | Message/log calibration gần nhất. |

---

### `Models/DetectionResult.cs`

Lưu lịch sử kết quả predict/anomaly detection.

Mỗi lần hệ thống detect một log sẽ sinh một record.

Các field chính:

| Field | Ý nghĩa |
|---|---|
| `RawLogId` | Log được detect. |
| `UserId` | User liên quan. |
| `BaselineRoute` | Route dùng khi predict: `global` hoặc `personalized`. |
| `Role`, `Department` | Context dùng cho calibration. |
| `Score` | Raw reconstruction score. |
| `Threshold` | Threshold được dùng để so sánh. |
| `GlobalRatio` | `score / tau_global`. |
| `RoleRatio` | `score / tau_role_or_department`. |
| `PersonalRatio` | `score / tau_personal`, chỉ có khi personalized. |
| `FinalAnomalyIndex` | Chỉ số anomaly cuối cùng sau khi combine global/role/personal ratio. |
| `IsAnomaly` | Có bị flag anomaly hay không. |
| `LogCountAtPrediction` | Log count tại thời điểm predict. |
| `ActiveDaysAtPrediction` | Active days tại thời điểm predict. |
| `PersonalizedReadyAtPrediction` | Lúc predict user đã có personalized chưa. |
| `PersonalizedTrainingTriggered` | Có trigger kiểm tra safe personalized calibration sau predict không. |
| `PersonalizedTrainingCompleted` | Personal threshold có được update trong lần kiểm tra đó không. |
| `ModelVersion` | Version/model metadata từ Python. |
| `Warning` | Cảnh báo nếu thiếu model, không đủ dữ liệu, v.v. |

---

## 6. Backend DTOs — dữ liệu vào/ra API

### `Dtos/LogInputDto.cs`

Payload để gọi `POST /api/logs` nhập một log thủ công.

Dùng khi muốn test một event mới từ Swagger hoặc frontend.

---

### `Dtos/BulkLogInputDto.cs`

Payload để gọi `POST /api/logs/bulk`.

Dùng để sinh/nhập nhanh nhiều log demo.

---

### `Dtos/DemoReplayRequest.cs`

Payload cho `POST /api/demo/replay`.

Các field:

| Field | Ý nghĩa |
|---|---|
| `UserId` | User holdout cần replay. |
| `Count` | Số ngày hoặc số event cần replay. |
| `Unit` | `days` hoặc `events`. Frontend hiện dùng `days`. |

---

### `Dtos/DetectionResponseDto.cs`

Response trả về sau khi detect.

Gồm toàn bộ thông tin cần hiển thị frontend:

- raw log id
- detection id
- user id
- log count
- active days
- route global/personalized
- role/department
- score/threshold
- global ratio/role ratio/personal ratio
- final anomaly index
- anomaly flag
- trạng thái train personalized

---

## 7. Backend Services — logic nghiệp vụ chính

### `Services/MlSettings.cs`

Class map cấu hình từ `appsettings.json` vào C# object.

Các service khác dùng class này để lấy:

- đường dẫn Python
- đường dẫn model
- số epoch
- timeout
- threshold quantile
- window size
- ngưỡng active days

---

### `Services/RuntimePaths.cs`

Chuẩn hóa và resolve đường dẫn runtime.

Vì backend chạy trong folder:

```text
backend/AnomalyFramework.Api
```

nhưng data/model/ml lại nằm ở root project, service này giúp convert các path tương đối như:

```text
../../data/anomaly.db
../../ml
../../models
```

thành absolute path đúng trên máy.

---

### `Services/PythonRunner.cs`

Service chịu trách nhiệm gọi Python script từ C#.

Nhiệm vụ:

- Tạo process Python.
- Truyền argument vào script.
- Set working directory là folder `ml`.
- Đọc `stdout`, `stderr`.
- Kiểm soát timeout.
- Trả về exit code và output.

Ví dụ backend gọi:

```text
python train_multiview.py --scope global ...
```

hoặc:

```text
python predict_multiview.py --user-id <USER_ID> --scope global ...
```

---

### `Services/MlJson.cs`

Class dùng để parse JSON output từ Python ML scripts.

Python scripts in kết quả dạng JSON ở cuối stdout, ví dụ:

```json
{
  "ok": true,
  "scope": "global",
  "threshold": 0.0123
}
```

Backend parse JSON này thành object C# để trả về API/frontend.

---

### `Services/TrainingService.cs`

Service quản lý train model.

Có hai luồng chính:

#### 1. Train global

Gọi script:

```text
ml/train_multiview.py --scope global
```

Input:

```text
data/processed/03_user_day_multiview.csv
```

Output:

```text
models/global/model.pt
models/global/vectorizer.joblib
models/global/metadata.json
models/global/all_scores.csv
```

#### 2. Fit/update safe personalized threshold

Gọi script:

```text
ml/train_multiview.py --scope personalized --user-id <USER_ID>
```

Input:

- RawLogs của user trong SQLite DB
- global model/vectorizer từ `models/global`

Output:

```text
models/personalized/<USER_ID>/metadata.json
models/personalized/<USER_ID>/calibration_scores.csv
```

Sau khi train xong, service update `UserModelStates`:

```text
IsPersonalizedReady = true
PersonalizedModelPath = ...
LastPersonalizedTrainingAtUtc = ...
```

---

### `Services/PredictionService.cs`

Service gọi Python để predict anomaly.

Nhiệm vụ:

- Nhận `userId` và `scope` (`global` hoặc `personalized`).
- Gọi script `ml/predict_multiview.py`.
- Parse output JSON.
- Trả về score, threshold, ratio, final index, anomaly flag.

---

### `Services/DetectionService.cs`

Service trung tâm của runtime detection.

Nó xử lý flow:

```text
Nhận log
↓
Lưu vào RawLogs
↓
Đếm logCount và activeDays của user
↓
Đọc UserModelState
↓
Nếu user có safe personalized threshold đang active → route personalized
Nếu chưa có → route global
↓
Gọi PredictionService để predict
↓
Lưu DetectionResult
↓
Nếu activeDays qua pre-check → kiểm tra đủ delayed safe days và cadence update
↓
Trả DetectionResponseDto
```

Điểm quan trọng: log hiện tại được predict bằng route hiện tại trước. Sau đó worker có thể kích hoạt/cập nhật personal threshold nếu đủ ngày an toàn, qua update delay và đến cadence. Neural model vẫn là global model.

---

### `Services/DemoReplayService.cs`

Service phục vụ demo với holdout users.

Nó đọc:

```text
data/processed/holdout/index.json
data/processed/holdout/<USER_ID>.csv
```

Sau đó replay log vào hệ thống.

Có 2 chế độ:

| Unit | Ý nghĩa |
|---|---|
| `days` | Replay đủ các event thuộc N active days tiếp theo. |
| `events` | Replay N event tiếp theo. |

Frontend dùng chế độ `days` để demo flow:

```text
Replay 1 active day
Replay to 30 active days
Replay 1 more day
```

---

## 8. Python ML — mô tả từng file

### `ml/requirements.txt`

Danh sách Python package cần cài cho ML worker.

Thường gồm:

```text
pandas
numpy
scikit-learn
joblib
torch
```

Cài bằng:

```powershell
cd ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

### `ml/model.py`

Định nghĩa kiến trúc model chính: **TCN-Transformer Autoencoder**.

Các class chính:

| Class | Ý nghĩa |
|---|---|
| `TCNBlock` | Một residual TCN block, dùng Conv1D dilation để học pattern theo thời gian. |
| `PositionalEncoding` | Thêm thông tin vị trí ngày trong window cho Transformer. |
| `TCNTransformerAutoencoder` | Model chính: input projection → TCN → Transformer Encoder → decoder reconstruction. |
| `DaySequenceEncoder` | Học trực tiếp token/source/time-gap/position của event trong ngày, rồi attention-pool về 64 chiều. |
| `MultiViewTCNTransformerAutoencoder` | Ghép count projection 64 chiều với neural sequence embedding 64 chiều trước temporal AE. |

Input model:

```text
Batch × 30 ngày × 128 chiều
```

Output model:

```text
Batch × 30 ngày × 128 chiều reconstructed
```

Model học reconstruct lại input. Anomaly score được tính bằng reconstruction error ở ngày cuối window.

---

### `ml/multiview_features.py`

Đây là file quan trọng nhất của pipeline mới. Nó chứa toàn bộ logic chuẩn hóa raw logs và tạo multi-view features.

Các nhóm chức năng chính:

#### 1. Đọc LDAP role context

Các hàm chính:

```text
load_ldap_context(...)
attach_role_context(...)
```

Nhiệm vụ:

- Đọc các file LDAP theo tháng trong `data/ldap`.
- Lấy các cột như `user_id`, `role`, `department`, `team`, `supervisor`.
- Gắn role/department vào event/user-day.

#### 2. Chuẩn hóa 5 raw logs thành unified event log

Các hàm:

```text
normalize_logon(...)
normalize_device(...)
normalize_file(...)
normalize_http(...)
normalize_email(...)
build_unified_event_log(...)
```

Mỗi raw log có schema khác nhau, nên file này map về schema chung:

```text
event_uid
source_file
original_id
timestamp
date
user
pc
event_type
object
object_type
hour
is_after_hours
is_weekend
event_order
source_payload
role
department
...
```

#### 3. Tạo count/statistical view

Hàm:

```text
build_user_day_features(...)
```

Output:

```text
data/processed/01_user_day_features.csv
```

Ví dụ feature:

```text
total_events
active_hours
n_logon
n_logoff
n_device_connect
n_file_events
n_http_events
n_email_sent
external_recipient_count
has_device_and_file_burst
z_file_vs_user_30d
z_usb_vs_user_30d
```

#### 4. Tạo raw sequence view

Hàm:

```text
build_user_day_sequences(...)
```

Output:

```text
data/processed/02_user_day_sequences.jsonl
```

File này lưu:

```text
event_sequence
source_sequence
time_gap_sequence
seq_len
sequence flags
```

Khi train, các trường raw này đi trực tiếp qua `DaySequenceEncoder`; các sequence-derived flags chỉ còn là metadata/feature phụ để audit.

#### 5. Join thành multiview

Hàm:

```text
build_user_day_multiview(...)
```

Output:

```text
data/processed/03_user_day_multiview.csv
```

Đây là input chính để train global model.

#### 6. Tạo window 30 ngày

Hàm:

```text
make_multiview_model_inputs(...)
```

Biến dữ liệu user-day thành input model:

```text
30 calendar days liên tục × 128 chiều; ngày không event được zero-fill và có empty-day sequence
```

#### 7. Runtime normalize từ SQLite RawLogs

Các hàm:

```text
read_events_from_sqlite(...)
normalize_runtime_rawlogs(...)
multiview_from_events(...)
```

Dùng cho personalized training và prediction runtime. Khi backend đã insert RawLogs vào SQLite, Python sẽ đọc lại RawLogs của user, chuẩn hóa thành events, rồi build lại count + sequence features.

---

### `ml/multiview_prepare_fast.py`

Script prepare dữ liệu mới theo multi-view framework.

Chạy bằng command:

```powershell
cd ml
python multiview_prepare_fast.py --cert-dir ../data/cert4.2 --ldap-dir ../data/ldap --out-dir ../data/processed --holdout-users 2 --min-active-days 30 --max-rows-per-file 200000 --sampling-mode time-user-stratified --holdout-strategy role-stratified
```

Nhiệm vụ:

1. Chọn complete user-month groups theo tháng và user trên đủ 5 nguồn; không lấy N dòng đầu theo mặc định.
2. Đọc LDAP monthly files.
3. Tạo unified event log.
4. Tạo count/statistical features.
5. Tạo raw token/source/time-gap sequences và sequence audit features.
6. Join thành user-day multiview.
7. Chọn holdout users có đủ lịch sử theo chiến lược role-stratified, không mặc định lấy top active users.
8. Ghi output vào `data/processed`.

Output chính:

| File | Ý nghĩa |
|---|---|
| `00_unified_event_log.csv` | Timeline chung của toàn bộ event sau khi chuẩn hóa. |
| `01_user_day_features.csv` | Count/statistical view theo user-day. |
| `02_user_day_sequences.jsonl` | Raw sequence + sequence audit fields theo user-day. |
| `03_user_day_multiview.csv` | File input chính để train global model. |
| `role_context.csv` | Role/department context từ LDAP. |
| `holdout/index.json` | Danh sách user demo được tách riêng. |
| `holdout/<USER_ID>.csv` | Log của từng holdout user để replay demo. |

---

### `ml/train_multiview.py`

Script train model mới.

Có 2 scope:

```text
global
personalized
```

#### Train global

Input:

```text
data/processed/03_user_day_multiview.csv
```

Command mẫu:

```powershell
python train_multiview.py --scope global --multiview-csv ../data/processed/03_user_day_multiview.csv --role-context ../data/processed/role_context.csv --feature-rules ./feature_rules.json --model-dir ../models/global --window-size 30 --epochs 1 --anomaly-quantile 0.995 --min-role-users 30 --min-role-user-days 1000
```

Nó làm:

1. Đọc multiview CSV.
2. Calendarize từng user timeline và zero-fill ngày trống.
3. Chia chronological train/validation/test theo user-day; không random windows.
4. Fit signed-log, RobustScaler và PCA count 64 chiều chỉ trên train rows.
5. Encode raw token/source/time-gap sequence thành 64 chiều bằng neural encoder.
6. Ghép thành 128 chiều và tạo rolling 30-calendar-day windows, stride 1.
7. Train TCN-Transformer Autoencoder trên train windows.
8. Fit global/role/department threshold trên validation windows.
9. Giữ test windows chỉ để đánh giá một lần.
10. Lưu model, vectorizer, metadata và `all_scores.csv`.

Output:

```text
models/global/model.pt
models/global/vectorizer.joblib
models/global/metadata.json
models/global/all_scores.csv
```

#### Fit/update safe personalized threshold

Input:

- SQLite DB `data/anomaly.db`
- `user_id`
- global model/vectorizer

Command mẫu:

```powershell
python train_multiview.py --scope personalized --db ../data/anomaly.db --user-id <USER_ID> --model-dir ../models/personalized/<USER_ID> --global-model-dir ../models/global --role-context ../data/processed/role_context.csv --feature-rules ./feature_rules.json --min-safe-days 30 --safe-history-days 90 --update-delay-days 7 --update-every-safe-days 7
```

Điểm quan trọng:

- Personalized dùng lại global model/vectorizer để score lịch sử của user.
- Chỉ ngày có score dưới role/backoff threshold, đã qua update delay và không ngay sau alert mới được coi là safe.
- Personal threshold dùng 90 ngày gần nhất, cần tối thiểu 30 safe days và chỉ update theo cadence.
- Không train neural model riêng cho từng user; thư mục personalized chỉ chứa threshold metadata và calibration scores.

---

### `ml/predict_multiview.py`

Script predict anomaly runtime.

Backend gọi script này mỗi khi cần detect user.

Input:

- SQLite DB `data/anomaly.db`
- user id
- scope: global hoặc personalized
- global model dir
- personalized threshold metadata dir nếu có

Nó làm:

1. Đọc RawLogs của user trong SQLite DB.
2. Chuẩn hóa thành unified events.
3. Build count view + sequence view.
4. Dùng global vectorizer để tạo vector 128 chiều.
5. Tạo latest 30-day window.
6. Luôn load global neural model; scope personalized chỉ thêm personal threshold.
7. Reconstruct window.
8. Tính reconstruction score cho ngày cuối window.
9. Tính `global_ratio`, `role_ratio`, `personal_ratio`.
10. Tính `final_anomaly_index`.
11. Trả JSON cho backend.

Công thức score hiện tại:

```text
e_count = MSE(64 chiều count thật, 64 chiều count reconstruct)
e_seq   = MSE(64 chiều sequence thật, 64 chiều sequence reconstruct)
score   = 0.5 * e_count + 0.5 * e_seq
```

Final anomaly index:

```text
Nếu chưa personalized:
final_index = 0.40 * global_ratio + 0.60 * role_ratio

Nếu đã personalized:
final_index = 0.25 * global_ratio + 0.35 * role_ratio + 0.40 * personal_ratio
```

Flag anomaly nếu:

```text
final_index > 1.0
```

---

## 9. Legacy ML files — file cũ còn giữ để tham khảo

Các file dưới đây thuộc flow cũ. Flow mới của project dùng nhóm file `multiview_*`.

### `ml/features.py`

File feature engineering cũ. Trước đây dùng để tạo feature/window đơn giản từ event log. Hiện không phải pipeline chính.

### `ml/train.py`

Script train model cũ. Trước đây train global/personalized bằng feature cũ. Hiện backend mới gọi `train_multiview.py`, không gọi file này.

### `ml/predict.py`

Script predict cũ. Hiện backend mới gọi `predict_multiview.py`, không gọi file này.

### `ml/cert_prepare.py`

Script prepare data cũ, đọc CSV full/in-memory nên dễ nặng với `http.csv`. Hiện nên dùng `multiview_prepare_fast.py`.

### `ml/README_ML.md`

README cũ cho phần ML cũ. Có thể giữ làm tài liệu tham khảo, nhưng README chính hiện tại nên theo multi-view.

---

## 10. Frontend — mô tả từng file

### `frontend/package.json`

Khai báo dependencies và scripts cho React/Vite.

Scripts:

| Script | Ý nghĩa |
|---|---|
| `npm run dev` | Chạy frontend dev server. |
| `npm run build` | Build frontend production. |
| `npm run preview` | Preview bản build. |

Dependencies chính:

- `react`
- `react-dom`
- `vite`
- `lucide-react`

---

### `frontend/index.html`

HTML entry point của frontend.

Vite sẽ mount React app vào div root trong file này.

---

### `frontend/src/App.jsx`

File chính của giao diện demo.

Nhiệm vụ:

- Gọi API backend.
- Load dashboard summary.
- Load holdout users.
- Cho chọn user demo.
- Replay user theo active days.
- Hiển thị detection result.
- Hiển thị recent detections.

Các nút demo chính:

| Nút | API gọi | Ý nghĩa |
|---|---|---|
| `Load holdout users` | `GET /api/demo/holdout-users` | Lấy user demo đã tách khỏi global training. |
| `Replay 1 active day` | `POST /api/demo/replay` | Replay 1 ngày hoạt động của user. |
| `Replay to 30 active days` | `POST /api/demo/replay` | Replay đủ 30 active days để trigger personalized training. |
| `Replay 1 more day` | `POST /api/demo/replay` | Replay thêm 1 ngày để thấy route personalized. |

Frontend hiển thị các thông tin:

- route: global/personalized
- role/department
- active days
- personalized ready
- score
- threshold
- global ratio
- role ratio
- personal ratio
- final anomaly index
- anomaly flag

---

### `frontend/src/styles.css`

CSS của UI demo.

Nhiệm vụ:

- Layout dashboard.
- Style card, button, table.
- Style trạng thái anomaly/normal.
- Làm giao diện dễ nhìn khi demo.

---

## 11. Data folder — dữ liệu vào/ra

### `data/cert4.2/`

Đặt 5 file raw CERT R4.2:

```text
logon.csv
device.csv
file.csv
email.csv
http.csv
```

Folder này thường rất nặng, không push lên GitHub.

---

### `data/ldap/`

Đặt các file LDAP theo tháng, ví dụ:

```text
2010-01.csv
2010-02.csv
...
2011-05.csv
```

LDAP cần các cột:

```text
employee_name,user_id,email,role,business_unit,functional_unit,department,team,supervisor
```

---

### `data/processed/`

Folder output sau khi chạy prepare.

Các file sinh ra:

| File | Ý nghĩa |
|---|---|
| `00_unified_event_log.csv` | Toàn bộ event sau chuẩn hóa. |
| `01_user_day_features.csv` | Count/statistical features theo user-day. |
| `02_user_day_sequences.jsonl` | Sequence-derived features và raw sequences theo user-day. |
| `03_user_day_multiview.csv` | Input chính để train global. |
| `role_context.csv` | Role context sau khi xử lý LDAP. |
| `prepare_summary.json` | Tóm tắt số dòng/user/holdout sau prepare nếu có. |
| `holdout/index.json` | Danh sách holdout users cho demo. |
| `holdout/<USER_ID>.csv` | Event của từng holdout user để replay. |

---

### `data/anomaly.db`

SQLite DB runtime, được backend tự tạo.

Chứa 3 bảng chính:

```text
RawLogs
UserModelStates
DetectionResults
```

File này là local runtime artifact, không push lên GitHub.

---

## 12. Models folder — model output

### `models/global/`

Output sau khi train global:

| File | Ý nghĩa |
|---|---|
| `model.pt` | PyTorch model weights cho global baseline. |
| `vectorizer.joblib` | Scaler/PCA fit trên train rows, dùng để chiếu count view về 64 chiều. |
| `metadata.json` | Threshold, role thresholds, department thresholds, model version, score formula. |
| `all_scores.csv` | Reconstruction score kèm chronological split/label metadata để audit và đánh giá. |

---

### `models/personalized/`

Mỗi user có calibration metadata riêng:

```text
models/personalized/<USER_ID>/metadata.json
models/personalized/<USER_ID>/calibration_scores.csv
```

Safe personal threshold chỉ active khi đủ delayed safe days; global neural model không được copy hoặc train lại theo user.

---

## 13. Cách chạy end-to-end

### Bước 1 — Clone repo

```powershell
git clone https://github.com/lethanhloi0603/ANOMALY-DETECTION.git
cd ANOMALY-DETECTION
```

---

### Bước 2 — Bỏ data vào đúng folder

Raw logs:

```text
data/cert4.2/logon.csv
data/cert4.2/device.csv
data/cert4.2/file.csv
data/cert4.2/email.csv
data/cert4.2/http.csv
```

LDAP:

```text
data/ldap/2010-01.csv
...
data/ldap/2011-05.csv
```

---

### Bước 3 — Setup Python

```powershell
cd ml
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Test:

```powershell
python -c "import pandas, joblib, torch; print('OK')"
```

---

### Bước 4 — Prepare multi-view data

Demo nhẹ:

```powershell
python multiview_prepare_fast.py --cert-dir ../data/cert4.2 --ldap-dir ../data/ldap --out-dir ../data/processed --holdout-users 2 --min-active-days 30 --max-rows-per-file 50000
```

Data nhiều hơn:

```powershell
python multiview_prepare_fast.py --cert-dir ../data/cert4.2 --ldap-dir ../data/ldap --out-dir ../data/processed --holdout-users 2 --min-active-days 30 --max-rows-per-file 200000
```

---

### Bước 5 — Train global bằng terminal để tránh Swagger timeout

```powershell
python train_multiview.py --scope global --multiview-csv ../data/processed/03_user_day_multiview.csv --role-context ../data/processed/role_context.csv --model-dir ../models/global --window-size 30 --epochs 1 --anomaly-quantile 0.995 --min-role-samples 25
```

Kiểm tra output:

```powershell
dir ..\models\global
```

Phải có:

```text
model.pt
vectorizer.joblib
metadata.json
train_scores.csv
```

---

### Bước 6 — Chạy backend

Mở terminal mới:

```powershell
cd backend\AnomalyFramework.Api
dotnet restore
dotnet run
```

Mở Swagger:

```text
http://localhost:5117/swagger
```

---

### Bước 7 — Chạy frontend

Mở terminal mới:

```powershell
cd frontend
npm install
npm run dev
```

Mở UI:

```text
http://localhost:5173
```

---

### Bước 8 — Demo flow

Trên frontend:

```text
1. Load holdout users
2. Chọn 1 user
3. Replay 1 active day
4. Replay to 30 active days
5. Replay 1 more day
```

Kỳ vọng:

```text
Replay 1 active day
→ route = global
→ có global ratio + role ratio

Replay to 30 active days
→ backend train personalized model
→ personalized ready = true

Replay 1 more day
→ route = personalized
→ có personal ratio + final anomaly index
```

---

## 14. Git/GitHub lưu ý

Không push các folder/file sau:

```text
data/cert4.2/
data/processed/
data/anomaly.db
models/global/model.pt
models/global/vectorizer.joblib
models/personalized/
ml/.venv/
frontend/node_modules/
```

Trước khi commit nên kiểm tra:

```powershell
git status --short
```

Nếu lỡ add data/model:

```powershell
git rm -r --cached data models ml/.venv frontend/node_modules
git add .gitignore
git commit -m "Remove generated data and model artifacts"
```

---

## 15. Điểm cần lưu ý về bản hiện tại

Bản hiện tại đã đồng bộ các lỗi kiến trúc chính với tài liệu: 30 calendar days, raw sequence encoder, past-only LDAP, validation-fitted thresholds, safe personalized threshold update và evaluation metrics. Khi dùng cho nghiên cứu chính thức vẫn cần:

1. Chạy `evaluate_multiview.py` cho nhiều quantile và đầy đủ A0–A5; không chốt tham số chỉ từ demo.
2. Regenerate `data/processed` bằng `full` hoặc `time-user-stratified`; các artifact cũ tạo bằng `head` vẫn bị lệch thời gian.
3. Chỉ bật after-hours trong `feature_rules.json` khi thí nghiệm công bố và kiểm chứng lịch làm việc; CERT R4.2 không cho một mốc 08:00–18:00 cố định toàn tổ chức.
4. Giữ hierarchy `Role → Department → Global`; chỉ mở rộng khi có đủ mẫu và ablation chứng minh hiệu quả.
5. AE-SAD không phải phần thiếu của model chính: framework chủ đích giữ unsupervised reconstruction, không dùng anomaly label trong loss.

---

## 16. Cách giải thích ngắn cho người mới vào project

Dự án này phát hiện anomaly bằng cách học hành vi theo cửa sổ 30 calendar days liên tục. Count view được chiếu về 64 chiều; raw token/source/time-gap sequence được neural encoder học về 64 chiều; hai nhánh ghép thành 128 chiều rồi đi qua TCN-Transformer Autoencoder. User mới dùng global threshold và backoff Role → Department → Global. Khi đủ delayed safe history, hệ thống chỉ fit/update personal threshold robust từ score của global model, không train neural model riêng cho user.
