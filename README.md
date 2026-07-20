# CERT R4.2 Insider Threat Anomaly Framework Demo

Project này dựng đúng luồng demo bạn mô tả:

- ASP.NET Core Web API nhận log, lưu SQLite, route baseline, gọi ML worker.
- Python/PyTorch worker train và predict Hybrid TCN-Transformer Autoencoder.
- React frontend cho nhập log thủ công, nhập nhanh bulk log, replay 2 user holdout.
- Khi user chưa đủ `50` raw logs: route qua `global`.
- Khi user đạt `>= 50` raw logs: backend tự train `personalized` model.
- Log tiếp theo sau khi train xong sẽ route qua `personalized`.

## 1. Prerequisites

Cài:

- .NET SDK 8
- Python 3.10 hoặc 3.11
- Node.js 18+
- VS Code

Kiểm tra:

```bash
dotnet --version
python --version
node --version
npm --version
```

## 2. Mở project

```bash
code cert-anomaly-framework
```

Cấu trúc:

```text
backend/AnomalyFramework.Api/  ASP.NET Core Web APIrontend/                      React Vite demo UI
ml/                            Python ML worker
data/cert4.2/                  bỏ CERT CSV gốc vào đây
data/processed/                output sau preprocess
models/                        model global/personalized
```

## 3. Bỏ CERT R4.2 CSV vào folder

Copy tối thiểu các file sau vào:

```text
data/cert4.2/
```

Gồm:

```text
logon.csv
device.csv
file.csv
email.csv
http.csv
```

## 4. Setup Python worker

Windows PowerShell:

```powershell
cd ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Preprocess và tách 2 user holdout

Đứng trong folder `ml`:

```bash
python cert_prepare.py --cert-dir ../data/cert4.2 --out-dir ../data/processed --holdout-users 2 --min-logs 50
```

Output:

```text
data/processed/global_events.csv
data/processed/holdout/index.json
data/processed/holdout/<USER_1>.csv
data/processed/holdout/<USER_2>.csv
```

Ý nghĩa:

- `global_events.csv`: train Global Baseline, đã loại 2 user holdout.
- `holdout/<USER>.csv`: dùng demo user mới nhập log từ frontend.

## 6. Chạy backend

Mở terminal mới ở root:

```bash
cd backend/AnomalyFramework.Api
dotnet restore
dotnet run
```

API:

```text
http://localhost:5117
```

Swagger:

```text
http://localhost:5117/swagger
```

SQLite DB tự tạo ở:

```text
data/anomaly.db
```

## 7. Train Global Baseline

Sau khi API chạy:

```bash
curl -X POST http://localhost:5117/api/training/global
```

Hoặc vào Swagger gọi `POST /api/training/global`.

Model global lưu ở:

```text
models/global/model.pt
models/global/scaler.joblib
models/global/metadata.json
```

## 8. Chạy frontend

Mở terminal mới:

```bash
cd frontend
npm install
npm run dev
```

Mở:

```text
http://localhost:5173
```

## 9. Demo flow theo đúng bài của bạn

### Cách A: Demo bằng 2 user holdout

1. Frontend → bấm `Load holdout users`.
2. Chọn 1 user.
3. Bấm `Replay 1 log`.

Kỳ vọng:

```text
Baseline route = global
Log count = 1
Personalized ready = false
```

4. Bấm `Replay 50 more logs`.

Kỳ vọng: backend thấy user đạt `>= 50 logs`, tự train personalized.

```text
Personalized training completed = true
Personalized ready = true
```

5. Bấm `Replay 1 more log`.

Kỳ vọng:

```text
Baseline route = personalized
```

### Cách B: Demo bằng log synthetic

1. Nhập user mới, ví dụ `U-DEMO-001`.
2. Bulk count = `51`.
3. Bấm `Generate & submit bulk logs`.
4. Sau khi training xong, submit thêm 1 log thủ công.
5. Log mới route qua personalized.

## 10. Config quan trọng

File:

```text
backend/AnomalyFramework.Api/appsettings.json
```

```json
{
  "Ml": {
    "PersonalizedThreshold": 50,
    "WindowSize": 30,
    "TrainEpochs": 3,
    "AnomalyQuantile": 0.95
  }
}
```

- `PersonalizedThreshold`: đủ bao nhiêu raw logs thì train personalized.
- `WindowSize`: 30 ngày.
- `TrainEpochs`: để 3 cho demo nhanh; tăng lên khi thực nghiệm thật.
- `AnomalyQuantile`: percentile của reconstruction error để lấy threshold.

## 11. API chính

### POST `/api/logs`

```json
{
  "userId": "U1234",
  "pcId": "PC-001",
  "timestamp": "2026-07-04T09:00:00Z",
  "eventType": "logon",
  "activity": "Logon",
  "content": "normal login"
}
```

### POST `/api/logs/bulk`

```json
{
  "userId": "U1234",
  "pcId": "PC-001",
  "count": 51,
  "scoreOnlyLast": true
}
```

### GET `/api/users/{userId}/status`

Xem user đủ log chưa, personalized ready chưa.

### GET `/api/detections?userId=U1234`

Xem detection history.

### POST `/api/training/global`

Train global baseline.

### POST `/api/training/personalized/{userId}`

Train personalized baseline thủ công.

### GET `/api/demo/holdout-users`

Đọc 2 user holdout đã tách.

### POST `/api/demo/replay`

```json
{
  "userId": "U1234",
  "count": 1
}
```

## 12. Ghi chú học thuật

Demo này ưu tiên chứng minh framework và routing. Để làm experiment học thuật hoàn chỉnh, bạn nên bổ sung thêm:

- Label từ `insiders.csv`.
- LDAP role/context encoder đầy đủ.
- Event sequence encoder riêng thay vì token-count theo ngày.
- Precision/Recall/F1/AUC.
- Split train/test theo user để tránh leakage.
- Background job queue thay vì train đồng bộ trong request.
