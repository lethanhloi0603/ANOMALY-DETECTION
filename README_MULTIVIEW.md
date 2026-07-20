# CERT R4.2 Multi-view Global–Role/Group–Personalized Framework

Bản này là bản điều chỉnh theo tài liệu BA mới:

- Không chỉ dùng count feature; mỗi user-day có 2 view: count/statistical view và sequence view.
- Global baseline có calibration theo Role/Group từ LDAP.
- Điều kiện chuyển personalized không còn là 50 logs, mà là `>= 30 active days`.
- Personalized baseline vẫn train riêng theo user, nhưng chỉ kích hoạt sau khi user có đủ 30 ngày hoạt động.

## 1. Bỏ dữ liệu vào folder

Raw logs:

```text
data/cert4.2/logon.csv
data/cert4.2/device.csv
data/cert4.2/file.csv
data/cert4.2/email.csv
data/cert4.2/http.csv
```

LDAP monthly files, ví dụ 17 file tháng:

```text
data/ldap/2010-01.csv
...
data/ldap/2011-05.csv
```

LDAP cần có các cột:

```text
employee_name,user_id,email,role,business_unit,functional_unit,department,team,supervisor
```

## 2. Setup Python

```powershell
cd C:\Users\ASUS\Downloads\cert-anomaly-framework\cert-anomaly-framework\ml
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu backend gọi sai Python, sửa `backend/AnomalyFramework.Api/appsettings.json`:

```json
"PythonExecutable": "C:/Users/ASUS/Downloads/cert-anomaly-framework/cert-anomaly-framework/ml/.venv/Scripts/python.exe"
```

## 3. Prepare multi-view data

Chạy sample nhanh trước:

```powershell
python multiview_prepare_fast.py --cert-dir ../data/cert4.2 --ldap-dir ../data/ldap --out-dir ../data/processed --holdout-users 2 --min-active-days 30 --max-rows-per-file 50000
```

Nếu không đủ user 30 ngày thì tăng:

```powershell
python multiview_prepare_fast.py --cert-dir ../data/cert4.2 --ldap-dir ../data/ldap --out-dir ../data/processed --holdout-users 2 --min-active-days 30 --max-rows-per-file 200000
```

Output mới:

```text
data/processed/00_unified_event_log.csv
data/processed/01_user_day_features.csv
data/processed/02_user_day_sequences.jsonl
data/processed/03_user_day_multiview.csv
data/processed/role_context.csv
data/processed/holdout/index.json
data/processed/holdout/<USER_ID>.csv
```

## 4. Reset DB/model personalized trước demo

```powershell
cd C:\Users\ASUS\Downloads\cert-anomaly-framework\cert-anomaly-framework
Remove-Item .\data\anomaly.db -ErrorAction SilentlyContinue
Remove-Item .\models\personalized\* -Recurse -Force -ErrorAction SilentlyContinue
```

## 5. Chạy backend

```powershell
cd C:\Users\ASUS\Downloads\cert-anomaly-framework\cert-anomaly-framework\backend\AnomalyFramework.Api
dotnet restore
dotnet run
```

Swagger:

```text
http://localhost:5117/swagger
```

## 6. Train Global Role/Group Baseline

Trong Swagger gọi:

```text
POST /api/training/global
```

Hoặc terminal:

```powershell
curl -X POST http://localhost:5117/api/training/global
```

Output model:

```text
models/global/model.pt
models/global/vectorizer.joblib
models/global/metadata.json
models/global/train_scores.csv
```

`metadata.json` có global threshold + role thresholds + department fallback thresholds.

## 7. Chạy frontend

Terminal mới:

```powershell
cd C:\Users\ASUS\Downloads\cert-anomaly-framework\cert-anomaly-framework\frontend
npm install
npm run dev
```

Mở:

```text
http://localhost:5173
```

## 8. Demo flow mới

1. Bấm `Load holdout users`.
2. Chọn user có role hiển thị.
3. Bấm `Replay 1 active day`.
   - Route: global.
   - Có role/department.
   - Có global ratio + role ratio.
4. Bấm `Replay to 30 active days`.
   - Backend tự train personalized vì user đủ 30 ngày hoạt động.
5. Bấm `Replay 1 more day`.
   - Route: personalized.
   - Có personal ratio và final anomaly index.

## 9. Điểm khác với bản cũ

Bản cũ:

```text
5 raw logs -> event stream -> count features -> 30-day window -> global/personal by 50 logs
```

Bản mới:

```text
5 raw logs + LDAP
  -> 00 unified_event_log
  -> 01 count/statistical features
  -> 02 behavior sequences
  -> 03 user_day_multiview with role context
  -> 30-day window, 128 dimensions = 64 count + 64 sequence
  -> TCN-Transformer Autoencoder
  -> Global + Role/Department calibration
  -> Personalized after 30 active days
```

## 10. Lưu ý học thuật

Bản này là demo framework chạy được. Nó đã có count + sequence + role/group calibration + personalized activation theo 30 active days. Để làm thực nghiệm học thuật đầy đủ, cần bổ sung label evaluation từ insiders.csv, chia train/val/test theo user, tính AUPRC/F1/Recall@K/detection delay và làm ablation A0-A5.
