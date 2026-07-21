# README_RUN_FROM_SCRATCH.md

# Hướng dẫn chạy project từ đầu sau khi pull/clone repo

Tài liệu này dành cho người mới pull code về máy và muốn chạy toàn bộ project **Anomaly Detection / Insider Threat Detection** từ đầu.

Project gồm 3 phần chính:

```text
backend/   → ASP.NET Core Web API, Swagger, SQLite DB, gọi Python ML worker
frontend/  → React UI để demo trực quan
ml/        → Python pipeline: prepare data, train global/personalized, predict
```

Lưu ý quan trọng:

```text
Repo GitHub KHÔNG chứa raw data, processed data, model trained, database local.
Người pull code về phải tự đặt data vào đúng folder rồi chạy prepare/train lại.
```

---

# 1. Yêu cầu cài sẵn trên máy

Trước khi chạy project, máy cần có:

## 1.1. Git

Kiểm tra:

```powershell
git --version
```

Nếu chưa có, cài Git for Windows.

## 1.2. Python 3.10 hoặc 3.11

Khuyến nghị dùng Python 3.10/3.11 vì PyTorch dễ cài hơn.

Kiểm tra:

```powershell
python --version
```

Nếu máy có nhiều Python version, dùng:

```powershell
py -0
```

## 1.3. .NET SDK 8

Kiểm tra:

```powershell
dotnet --version
```

Nếu chưa có, cài .NET 8 SDK.

## 1.4. Node.js LTS

Kiểm tra:

```powershell
node -v
npm -v
```

Nếu chưa có, cài Node.js bản LTS.

---

# 2. Clone code từ GitHub

Mở PowerShell, vào folder muốn chứa source code:

```powershell
cd C:\Users\<TEN_MAY>\Downloads
```

Clone repo:

```powershell
git clone https://github.com/lethanhloi0603/ANOMALY-DETECTION.git
```

Vào project:

```powershell
cd ANOMALY-DETECTION
```

Kiểm tra đúng root project:

```powershell
dir
```

Phải thấy các folder như:

```text
backend
frontend
ml
data
models
```

---

# 3. Chuẩn bị data

Do repo không push data, cần tự copy data vào máy.

## 3.1. Raw CERT R4.2 logs

Tạo hoặc kiểm tra folder:

```text
data/cert4.2/
```

Bỏ các file CERT vào đây:

```text
data/cert4.2/logon.csv
data/cert4.2/device.csv
data/cert4.2/file.csv
data/cert4.2/email.csv
data/cert4.2/http.csv
```

`http.csv` có thể rất lớn. Điều này bình thường.

## 3.2. LDAP monthly files

Tạo hoặc kiểm tra folder:

```text
data/ldap/
```

Bỏ các file LDAP theo tháng vào đây, ví dụ:

```text
data/ldap/2010-01.csv
data/ldap/2010-02.csv
...
data/ldap/2011-05.csv
```

LDAP file nên có các cột dạng:

```text
employee_name,user_id,email,role,business_unit,functional_unit,department,team,supervisor
```

## 3.3. Những folder không cần copy

Các folder/file sau sẽ được generate sau, không cần copy từ GitHub:

```text
data/processed/
data/anomaly.db
models/global/
models/personalized/
```

---

# 4. Setup Python ML environment

Vào folder `ml`:

```powershell
cd C:\Users\<TEN_MAY>\Downloads\ANOMALY-DETECTION\ml
```

Tạo virtual environment:

```powershell
python -m venv .venv
```

Cho phép chạy script trong PowerShell hiện tại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Active venv:

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu active thành công, đầu dòng terminal sẽ có:

```text
(.venv) PS ...
```

Cài thư viện:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Test package:

```powershell
python -c "import pandas, joblib, torch; print('OK')"
```

Nếu in ra `OK` là xong phần Python.

---

# 5. Cấu hình backend dùng đúng Python trong `.venv`

Backend cần biết phải gọi Python nào để chạy script ML.

Trong folder `ml`, chạy:

```powershell
python -c "import sys; print(sys.executable)"
```

Nó sẽ in ra đường dẫn kiểu:

```text
C:\Users\<TEN_MAY>\Downloads\ANOMALY-DETECTION\ml\.venv\Scripts\python.exe
```

Copy đường dẫn đó.

Tạo file config local:

```powershell
notepad ..\backend\AnomalyFramework.Api\appsettings.Development.json
```

Dán nội dung sau, nhớ đổi path theo máy mình:

```json
{
  "Ml": {
    "PythonExecutable": "C:/Users/<TEN_MAY>/Downloads/ANOMALY-DETECTION/ml/.venv/Scripts/python.exe",
    "TrainEpochs": 1,
    "TrainingTimeoutSeconds": 3600,
    "PredictionTimeoutSeconds": 120
  }
}
```

Lưu ý:

```text
Dùng dấu / trong path cho dễ chạy.
File appsettings.Development.json không được push lên GitHub vì đây là config local của từng máy.
```

---

# 6. Prepare multi-view data

Vẫn đang ở folder:

```text
ANOMALY-DETECTION/ml
```

Chạy prepare data.

## 6.1. Bản nhẹ để test nhanh

```powershell
python multiview_prepare_fast.py --cert-dir ../data/cert4.2 --ldap-dir ../data/ldap --out-dir ../data/processed --holdout-users 2 --min-active-days 30 --max-rows-per-file 50000
```

## 6.2. Bản lớn hơn nếu máy đủ khỏe

```powershell
python multiview_prepare_fast.py --cert-dir ../data/cert4.2 --ldap-dir ../data/ldap --out-dir ../data/processed --holdout-users 2 --min-active-days 30 --max-rows-per-file 200000
```

Nếu chạy thành công, sẽ có các file:

```text
data/processed/00_unified_event_log.csv
data/processed/01_user_day_features.csv
data/processed/02_user_day_sequences.jsonl
data/processed/03_user_day_multiview.csv
data/processed/role_context.csv
data/processed/holdout/index.json
data/processed/holdout/<USER_ID>.csv
```

Ý nghĩa nhanh:

```text
00_unified_event_log.csv
→ Timeline event chung sau khi chuẩn hóa 5 log CERT.

01_user_day_features.csv
→ Count/statistical view theo user-day.

02_user_day_sequences.jsonl
→ Sequence view theo user-day.

03_user_day_multiview.csv
→ Input chính để train global baseline.

role_context.csv
→ Role/department context từ LDAP.

holdout/
→ User demo được tách riêng để replay.
```

---

# 7. Reset DB và model cá nhân hóa

Trước khi demo từ đầu, nên reset local DB và personalized model.

Về root project:

```powershell
cd ..
```

Bây giờ đang ở:

```text
ANOMALY-DETECTION
```

Chạy:

```powershell
Remove-Item .\data\anomaly.db -ErrorAction SilentlyContinue
Remove-Item .\models\personalized\* -Recurse -Force -ErrorAction SilentlyContinue
```

Không cần xóa `data/processed` nếu prepare đã chạy xong.

---

# 8. Train Global Baseline

Có 2 cách train: train trực tiếp bằng terminal hoặc train qua Swagger.

Khuyến nghị dùng **terminal** vì tránh timeout Swagger.

## 8.1. Train global bằng terminal

Vào folder `ml`:

```powershell
cd .\ml
.\.venv\Scripts\Activate.ps1
```

Chạy:

```powershell
python train_multiview.py --scope global --multiview-csv ../data/processed/03_user_day_multiview.csv --role-context ../data/processed/role_context.csv --model-dir ../models/global --window-size 30 --epochs 1 --anomaly-quantile 0.995 --min-role-samples 25
```

Nếu thành công sẽ in JSON có:

```json
{
  "ok": true,
  "scope": "global"
}
```

Kiểm tra model:

```powershell
dir ..\models\global
```

Phải thấy:

```text
model.pt
vectorizer.joblib
metadata.json
train_scores.csv
```

## 8.2. Train global bằng Swagger

Chỉ dùng khi backend đã chạy.

Gọi:

```text
POST /api/training/global
```

Nếu dữ liệu lớn, Swagger có thể timeout. Khi đó dùng cách terminal ở trên.

---

# 9. Chạy backend API + Swagger

Mở PowerShell mới:

```powershell
cd C:\Users\<TEN_MAY>\Downloads\ANOMALY-DETECTION\backend\AnomalyFramework.Api
dotnet restore
dotnet run
```

Nếu chạy thành công sẽ thấy:

```text
Now listening on: http://localhost:5117
```

Mở Swagger:

```text
http://localhost:5117/swagger
```

Một số API quan trọng:

```text
POST /api/training/global
→ Train global baseline.

GET /api/demo/holdout-users
→ Lấy danh sách user demo.

POST /api/demo/replay
→ Replay log/day của holdout user vào DB.

GET /api/detections
→ Xem lịch sử detection.

GET /api/dashboard/summary
→ Xem thống kê dashboard.
```

---

# 10. Chạy frontend React

Mở PowerShell mới:

```powershell
cd C:\Users\<TEN_MAY>\Downloads\ANOMALY-DETECTION\frontend
npm install
npm run dev
```

Nếu chạy thành công sẽ thấy URL dạng:

```text
http://localhost:5173
```

Mở browser:

```text
http://localhost:5173
```

---

# 11. Demo flow trên giao diện

Trên frontend làm theo thứ tự:

```text
1. Bấm Load holdout users
2. Chọn một user
3. Bấm Replay 1 active day
4. Xem kết quả route = global / role
5. Bấm Replay to 30 active days
6. Hệ thống trigger train personalized model
7. Bấm Replay 1 more day
8. Xem kết quả route = personalized
```

Kỳ vọng:

```text
User mới
→ dùng Global + Role/Department calibration

User đủ 30 active days
→ train Personalized Baseline

Log/ngày tiếp theo
→ route qua Personalized Baseline
```

---

# 12. Flow tổng thể của hệ thống

```text
Raw CERT logs + LDAP
        ↓
ml/multiview_prepare_fast.py
        ↓
00_unified_event_log.csv
01_user_day_features.csv
02_user_day_sequences.jsonl
03_user_day_multiview.csv
        ↓
ml/train_multiview.py
        ↓
models/global/model.pt
models/global/vectorizer.joblib
models/global/metadata.json
        ↓
backend API
        ↓
frontend demo
        ↓
Replay holdout user
        ↓
Global / Role calibration
        ↓
30 active days
        ↓
Personalized baseline
```

---

# 13. Các lỗi thường gặp và cách xử lý

## 13.1. Lỗi `No module named pandas/joblib/torch`

Nguyên nhân: chưa cài package trong `.venv`, hoặc backend gọi nhầm Python.

Cách xử lý:

```powershell
cd C:\Users\<TEN_MAY>\Downloads\ANOMALY-DETECTION\ml
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import pandas, joblib, torch; print('OK')"
```

Sau đó kiểm tra `appsettings.Development.json` đã trỏ đúng `.venv/Scripts/python.exe` chưa.

## 13.2. Lỗi không tìm thấy folder `ml`

Nguyên nhân: đang đứng sai path.

Cách tìm đúng folder:

```powershell
cd C:\Users\<TEN_MAY>\Downloads
Get-ChildItem -Recurse -Directory -Filter ml | Select-Object FullName
```

Copy đúng path có `multiview_prepare_fast.py`.

## 13.3. Lỗi timeout khi train global qua Swagger

Nguyên nhân: train bằng API lâu hơn timeout.

Cách xử lý: train trực tiếp bằng terminal:

```powershell
cd C:\Users\<TEN_MAY>\Downloads\ANOMALY-DETECTION\ml
.\.venv\Scripts\Activate.ps1
python train_multiview.py --scope global --multiview-csv ../data/processed/03_user_day_multiview.csv --role-context ../data/processed/role_context.csv --model-dir ../models/global --window-size 30 --epochs 1 --anomaly-quantile 0.995 --min-role-samples 25
```

## 13.4. Lỗi thiếu file CERT CSV

Kiểm tra:

```powershell
dir C:\Users\<TEN_MAY>\Downloads\ANOMALY-DETECTION\data\cert4.2
```

Phải có:

```text
logon.csv
device.csv
file.csv
email.csv
http.csv
```

## 13.5. Lỗi frontend không gọi được API

Đảm bảo backend đang chạy ở:

```text
http://localhost:5117
```

và frontend mở ở:

```text
http://localhost:5173
```

Nếu đổi IP/LAN thì cần chỉnh API URL trong `frontend/src/App.jsx`.

---

# 14. Lưu ý khi làm việc với GitHub

Không push các folder/file sau:

```text
data/cert4.2/
data/processed/
data/ldap/*.csv
data/anomaly.db
models/global/
models/personalized/
ml/.venv/
frontend/node_modules/
```

Trước khi commit luôn chạy:

```powershell
git status --short
```

Không được thấy file data/model lớn.

Quy trình làm việc đề xuất:

```powershell
git checkout main
git pull origin main
git checkout -b feature/ten-task

# code...

git add .
git commit -m "Mo ta thay doi"
git push -u origin feature/ten-task
```

Sau đó tạo Pull Request vào `main`.

---

# 15. Checklist chạy từ đầu

Dùng checklist này để tự kiểm tra:

```text
[ ] Clone repo về máy
[ ] Copy CERT CSV vào data/cert4.2
[ ] Copy LDAP monthly CSV vào data/ldap
[ ] Tạo Python .venv trong ml
[ ] pip install -r requirements.txt
[ ] Tạo appsettings.Development.json trỏ đúng PythonExecutable
[ ] Chạy multiview_prepare_fast.py thành công
[ ] Có data/processed/03_user_day_multiview.csv
[ ] Reset data/anomaly.db và models/personalized
[ ] Train global thành công
[ ] Có models/global/model.pt
[ ] Chạy backend dotnet run
[ ] Mở Swagger được
[ ] Chạy frontend npm run dev
[ ] Load holdout users được
[ ] Replay 1 active day được
[ ] Replay to 30 active days được
[ ] Replay 1 more day chuyển personalized được
```

---

# 16. Ghi chú về bản hiện tại

Bản hiện tại đã có:

```text
Count/statistical view
Sequence-derived feature view
Global baseline
Role/Department calibration
Basic personalized baseline sau 30 active days
Frontend demo trực quan
```

Bản hiện tại chưa phải full research production:

```text
Safe personalized calibration đầy đủ chưa implement
Sequence view chưa phải token embedding encoder trực tiếp
AE-SAD semi-supervised loss chưa implement
Evaluation metrics script chưa có
```

Các phần này là hướng mở rộng tiếp theo.
