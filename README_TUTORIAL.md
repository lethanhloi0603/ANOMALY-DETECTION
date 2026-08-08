# Hướng dẫn cài đặt và chạy Insider Threat Framework

Tài liệu này hướng dẫn một người mới tải source về, chuẩn bị môi trường Windows,
gắn dataset CERT R4.2, chạy kiểm tra và chạy thực nghiệm đầy đủ.

Các lệnh bên dưới sử dụng PowerShell và giả sử repository nằm tại:

```text
C:\Users\<USER>\Downloads\Insider threat
```

Thay đường dẫn cho phù hợp với máy của người chạy.

## 1. Yêu cầu máy

### Phần mềm

- Windows 10/11 x64.
- Python 3.11 x64.
- PowerShell.
- Kết nối Internet khi cài dependency.
- Git nếu tải source bằng Git.
- Docker Desktop chỉ cần khi muốn chạy backend bằng PostgreSQL/Docker.

### Phần cứng tối thiểu

- RAM: 8 GB.
- CPU x64, khuyến nghị từ 4 core.
- Ổ đĩa trống: tối thiểu 30–50 GB ngoài dung lượng dataset.
- Dataset CERT R4.2 khoảng 15 GB sau giải nén.

Với máy RAM 8 GB:

- Dùng `batch-size=2`.
- Đóng browser, Docker, IDE nặng trước khi chạy full.
- Cắm sạc và tắt Sleep.
- Lần đầu dùng `bootstrap=0`.

## 2. Tải source

### Cách A — dùng Git

```powershell
Set-Location "C:\Users\<USER>\Downloads"
git clone <REPOSITORY_URL> "Insider threat"
Set-Location "C:\Users\<USER>\Downloads\Insider threat"
```

Thay `<REPOSITORY_URL>` bằng URL repository thật.

### Cách B — tải ZIP

1. Tải ZIP của source.
2. Extract thành thư mục `Insider threat`.
3. Mở PowerShell.
4. Chuyển vào thư mục:

```powershell
Set-Location "C:\Users\<USER>\Downloads\Insider threat"
```

Kiểm tra:

```powershell
Get-ChildItem -Force
```

Phải thấy ít nhất:

```text
backend
machine_learning
data
docs
compose.yaml
README_SYSTEM.md
README_TUTORIAL.md
```

## 3. Cài Python 3.11

Tải Python 3.11 x64 từ trang chính thức của Python. Khi cài:

- Chọn bản 64-bit.
- Chọn `Add Python to PATH`.
- Có thể chọn `Install launcher for all users`.

Kiểm tra:

```powershell
py -3.11 --version
```

Kết quả phải là Python 3.11.x.

Không khuyến nghị Python 3.10 hoặc 3.12 cho môi trường này vì `pyproject.toml` đang
khóa `requires-python >=3.11` và workspace hiện được kiểm thử bằng Python 3.11.

## 4. Tạo virtual environment

Tại thư mục gốc repository:

```powershell
Set-Location "C:\Users\<USER>\Downloads\Insider threat"
py -3.11 -m venv .venv
```

Không bắt buộc activate environment. Các hướng dẫn sau gọi trực tiếp Python trong
`.venv`, tránh lỗi PowerShell ExecutionPolicy.

Nâng cấp công cụ cài package:

```powershell
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

## 5. Cài PyTorch CPU và dependency

Máy không có NVIDIA CUDA nên cài PyTorch CPU:

```powershell
& .\.venv\Scripts\python.exe -m pip install torch `
  --index-url https://download.pytorch.org/whl/cpu
```

Cài package machine learning và dependency test:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".\machine_learning[dev]"
```

Cài backend và dependency test:

```powershell
& .\.venv\Scripts\python.exe -m pip install -e ".\backend[dev]"
```

Kiểm tra import:

```powershell
& .\.venv\Scripts\python.exe -c "import torch, numpy, pyarrow; print(torch.__version__); print(torch.cuda.is_available())"
```

Trên máy CPU, dòng cuối phải là:

```text
False
```

## 6. Chuẩn bị dataset CERT R4.2

Dataset không được đóng gói trong repository. Người chạy phải có quyền sử dụng và tự
tải CERT R4.2 từ nguồn được phép, sau đó giải nén ra một thư mục riêng.

Ví dụ:

```text
C:\Users\<USER>\Downloads\cert4.2
```

Cấu trúc bắt buộc:

```text
cert4.2/
├─ logon.csv
├─ device.csv
├─ file.csv
├─ http.csv
├─ email.csv
├─ LDAP/
│  ├─ 2009-12.csv
│  ├─ 2010-01.csv
│  └─ ...
└─ answers/
   ├─ insiders.csv
   ├─ r2.csv
   └─ ...
```

Các file sau có thể tồn tại nhưng không được dùng trong primary model:

```text
psychometric.csv
insiders.csv ở root
```

`psychometric.csv`/OCEAN chỉ dành cho ablation. `answers/` chỉ được đọc ở evaluation.

Kiểm tra dataset:

```powershell
$certRoot = "C:\Users\<USER>\Downloads\cert4.2"

Get-Item `
  "$certRoot\logon.csv", `
  "$certRoot\device.csv", `
  "$certRoot\file.csv", `
  "$certRoot\http.csv", `
  "$certRoot\email.csv"

Get-ChildItem "$certRoot\LDAP" -Filter "*.csv" | Select-Object -First 5 Name
Get-ChildItem "$certRoot\answers" -Filter "*.csv" | Select-Object -First 5 Name
```

Nếu một trong năm CSV chính hoặc folder `LDAP` bị thiếu, không chạy pipeline.

## 7. Gắn dataset vào `data/raw` mà không copy

Không copy 15 GB dataset vào repository. Tạo Windows directory junction:

```powershell
Set-Location "C:\Users\<USER>\Downloads\Insider threat"

$certSource = "C:\Users\<USER>\Downloads\cert4.2"
$rawParent = Join-Path (Get-Location) "data\raw"
$rawLink = Join-Path $rawParent "cert4.2"

New-Item -ItemType Directory -Path $rawParent -Force | Out-Null

if (Test-Path -LiteralPath $rawLink) {
    throw "data\raw\cert4.2 đã tồn tại. Hãy kiểm tra trước, không ghi đè."
}

New-Item -ItemType Junction -Path $rawLink -Target $certSource
```

Kiểm tra junction:

```powershell
Get-Item -LiteralPath ".\data\raw\cert4.2" |
  Select-Object FullName, LinkType, Target
```

`LinkType` phải là `Junction`, còn `Target` phải trỏ tới folder dataset thật.

Junction không tạo bản sao dữ liệu. Xóa junction không xóa dataset đích, nhưng luôn
kiểm tra `Target` trước khi thực hiện bất kỳ thao tác xóa nào.

## 8. Chạy test trước khi chạy dữ liệu

### Machine learning

```powershell
Set-Location "C:\Users\<USER>\Downloads\Insider threat\machine_learning"
..\.venv\Scripts\python.exe -m pytest
..\.venv\Scripts\python.exe -m ruff check .
```

### Backend

```powershell
Set-Location "C:\Users\<USER>\Downloads\Insider threat\backend"
..\.venv\Scripts\python.exe -m pytest
..\.venv\Scripts\python.exe -m ruff check .
```

Không chạy full experiment nếu test thất bại.

## 9. Chạy smoke end-to-end

Smoke dùng dữ liệu CERT thật nhưng chỉ đọc tối đa 10.000 dòng mỗi source và train một
epoch. Nó kiểm tra đường chạy, không tạo metric nghiên cứu.

```powershell
Set-Location "C:\Users\<USER>\Downloads\Insider threat\machine_learning"
..\.venv\Scripts\python.exe -m cli.run_framework
```

Artifact dự kiến:

```text
data/processed/smoke/user_days.sqlite
data/processed/smoke/user_days.sqlite.manifest.json
data/artifacts/smoke/low_ram_tcn_transformer_ae.v4.pt
data/artifacts/smoke/low_ram_references.train.json
data/artifacts/smoke/low_ram_train_predictions.csv
data/artifacts/smoke/smoke_report.json
```

Kiểm tra:

```powershell
Get-Content -Raw `
  "C:\Users\<USER>\Downloads\Insider threat\data\artifacts\smoke\smoke_report.json"
```

`status` phải là `PASS`.

Smoke có thể có `calibrated_scores=0` vì shard hai ngày không đủ support
Person/Role/Global. Đây không phải lỗi.

## 10. Chỉ build full store

Nên chạy bước này trước để xác nhận raw parsing và đo thời gian:

```powershell
Set-Location "C:\Users\<USER>\Downloads\Insider threat\machine_learning"

..\.venv\Scripts\python.exe -m cli.build_store `
  --raw-root "..\data\raw\cert4.2" `
  --store "..\data\processed\cert4.2_user_days.sqlite"
```

Không truyền `--max-rows-per-source` hoặc `--end-day` khi build full.

Output:

```text
data/processed/cert4.2_user_days.sqlite
data/processed/cert4.2_user_days.sqlite.manifest.json
```

Build store có checkpoint theo source. Nếu bị dừng, chạy lại đúng lệnh; source có
fingerprint hợp lệ sẽ được bỏ qua.

## 11. Chạy full experiment trên máy RAM 8 GB

Trước khi chạy:

1. Khởi động lại Windows.
2. Đóng trình duyệt, Docker Desktop và ứng dụng nặng.
3. Cắm sạc.
4. Tắt Sleep khi đang cắm sạc.
5. Kiểm tra còn ít nhất khoảng 3 GB RAM available.
6. Không chạy backend server cùng lúc.

Lần đầu chạy không bootstrap để sớm có metric:

```powershell
Set-Location "C:\Users\<USER>\Downloads\Insider threat\machine_learning"

..\.venv\Scripts\python.exe -m cli.run_experiment `
  --batch-size 2 `
  --epochs 20 `
  --bootstrap 0 `
  --device cpu
```

`run_experiment` sẽ tự thực hiện:

```text
build/resume store
→ train/resume model
→ score toàn bộ Train
→ fit frozen Train references
→ score Validation
→ tạo Validation universe/labels
→ chọn threshold q=0.995
→ score Test
→ tạo Test universe/labels
→ tính Test metrics bằng locked threshold
→ ghi experiment_report.json
```

Với Intel i5-8250U và RAM 8 GB, full run không bootstrap có thể mất khoảng 2–5
ngày chạy liên tục. Thời gian thực tế phụ thuộc nhiệt độ, tốc độ SSD và lượng RAM trống.

## 12. Chạy bootstrap sau khi đã có metric

Sau khi full run `bootstrap=0` hoàn thành, chạy lại:

```powershell
Set-Location "C:\Users\<USER>\Downloads\Insider threat\machine_learning"

..\.venv\Scripts\python.exe -m cli.run_experiment `
  --batch-size 2 `
  --epochs 20 `
  --bootstrap 1000 `
  --device cpu
```

Resume mặc định đang bật. Store, checkpoint, reference và prediction có checksum hợp
lệ sẽ được tái sử dụng; evaluator sẽ tính thêm confidence interval.

## 13. Kết quả full experiment

Thư mục chính:

```text
data/artifacts/experiment_v1/
```

Các file quan trọng:

| File | Ý nghĩa |
|---|---|
| `tcn_transformer_ae.v4.pt` | Model checkpoint |
| `tcn_transformer_ae.v4.pt.manifest.json` | Checksum và lịch sử train |
| `train_raw_scores.csv` | Reconstruction error trên Train |
| `references.train.json` | Frozen Person/Role/Global empirical references |
| `validation_predictions.csv` | Validation calibrated risk |
| `validation.metrics.json` | Threshold và Validation metric |
| `test_predictions.csv` | Test calibrated risk |
| `test.metrics.json` | Test metric với threshold đã khóa |
| `experiment_report.json` | Báo cáo tóm tắt toàn experiment |

Universe và label được đặt riêng tại:

```text
data/evaluation/experiment_v1/
```

## 14. Resume khi máy bị dừng

Chạy lại đúng lệnh full:

```powershell
..\.venv\Scripts\python.exe -m cli.run_experiment `
  --batch-size 2 `
  --epochs 20 `
  --bootstrap 0 `
  --device cpu
```

Không thay đổi:

- Dataset source.
- Feature/Sequence/model config.
- Batch-independent preprocessing.
- Store path.
- Checkpoint path.
- Learning rate.
- Endpoint policy.

Nếu checksum thay đổi, pipeline sẽ từ chối artifact cũ. Không sửa manifest bằng tay.

## 15. Theo dõi tiến trình

Mở Task Manager:

- `Performance → Memory`: tránh để memory liên tục ở 100%.
- `Performance → CPU`: CPU cao trong train/score là bình thường.
- `Performance → Disk`: nếu disk luôn 100% và CPU thấp, máy đang paging hoặc bị I/O
  bottleneck.

Kiểm tra artifact đang được cập nhật:

```powershell
Get-ChildItem `
  "C:\Users\<USER>\Downloads\Insider threat\data\processed", `
  "C:\Users\<USER>\Downloads\Insider threat\data\artifacts\experiment_v1" `
  -Recurse -File -ErrorAction SilentlyContinue |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 15 FullName, Length, LastWriteTime
```

## 16. Xử lý lỗi thường gặp

### `Python 3.11 was not found`

Kiểm tra:

```powershell
py --list
```

Cài Python 3.11 x64 rồi tạo lại `.venv`.

### `ModuleNotFoundError: insider_ml`

Đang chạy sai environment hoặc chưa cài ML package:

```powershell
Set-Location "C:\Users\<USER>\Downloads\Insider threat"
& .\.venv\Scripts\python.exe -m pip install -e ".\machine_learning[dev]"
```

### `DLL load failed` khi import Torch/Numpy

- Xác nhận Python là x64.
- Cài Microsoft Visual C++ Redistributable x64.
- Xóa và tạo lại `.venv` nếu đã trộn package của nhiều Python version.

### Không tìm thấy raw CSV

Kiểm tra junction:

```powershell
Get-Item ".\data\raw\cert4.2" | Select-Object LinkType, Target
Get-ChildItem ".\data\raw\cert4.2"
```

### `full experiment refuses capped/incomplete source`

Store đã được tạo bằng smoke cap. Không dùng smoke store cho full experiment. Full
store mặc định phải là:

```text
data/processed/cert4.2_user_days.sqlite
```

### `full experiment requires store through locked Test end`

Store kết thúc trước `2011-05-17`. Build lại full mà không truyền `--end-day`.

### `no global branch can calibrate`

Raw source hoặc LDAP universe chưa đầy đủ, hoặc store được build bằng giới hạn smoke.
Đọc file manifest cạnh SQLite store và kiểm tra `readiness_preflight`.

### Máy thiếu RAM hoặc pagefile tăng mạnh

1. Dừng tiến trình bằng `Ctrl+C`.
2. Khởi động lại máy.
3. Đóng ứng dụng nền.
4. Chạy lại với `--batch-size 1`.

Store/checkpoint đã hoàn thành vẫn có thể resume.

### Validation có zero calibrated scores

Đọc:

```text
data/processed/cert4.2_user_days.sqlite.manifest.json
data/artifacts/experiment_v1/references.train.json
```

Kiểm tra support Global Feature/Sequence. Pipeline cố ý dừng thay vì tạo metric trên
prediction rỗng.

## 17. Chạy backend local bằng SQLite

Backend không bắt buộc để chạy research experiment.

File `.env.example` là tài liệu mẫu; backend hiện đọc trực tiếp biến môi trường của
process và không tự load file `.env`. Với development local, đặt biến trong cửa sổ
PowerShell hiện tại:

```powershell
Set-Location "C:\Users\<USER>\Downloads\Insider threat\backend"

$env:APP_ENV = "development"
$env:DATABASE_URL = "sqlite:///../data/runtime/insider_threat.db"
$env:API_KEY = "local-api-key"
$env:SCORER_API_KEY = "local-scorer-key"
```

Nếu đã chạy Validation, nạp đúng locked threshold từ báo cáo:

```powershell
$validationReport = Get-Content -Raw `
  "..\data\artifacts\experiment_v1\validation.metrics.json" |
  ConvertFrom-Json

$env:LOCKED_ALERT_THRESHOLD = [string]$validationReport.threshold
```

Nếu chỉ kiểm tra health/API contract và chưa score assessment thì development có thể
để trống `LOCKED_ALERT_THRESHOLD`.

Khởi tạo và chạy:

```powershell
..\.venv\Scripts\python.exe -m alembic upgrade head
..\.venv\Scripts\python.exe -m cli.init_db
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

API mặc định:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/docs
```

Không dùng backend production để thay thế `run_experiment`; hai luồng có mục đích khác
nhau.

## 18. Chạy backend bằng Docker

Docker chỉ phục vụ backend/PostgreSQL, không làm full ML experiment nhanh hơn.

Tại thư mục gốc:

```powershell
$validationReport = Get-Content -Raw `
  ".\data\artifacts\experiment_v1\validation.metrics.json" |
  ConvertFrom-Json

$env:API_KEY = "replace-with-a-long-secret"
$env:SCORER_API_KEY = "replace-with-another-long-secret"
$env:LOCKED_ALERT_THRESHOLD = [string]$validationReport.threshold

docker compose up --build
```

`API_KEY` và `SCORER_API_KEY` phải khác nhau. Trong triển khai thật,
`LOCKED_ALERT_THRESHOLD` phải là giá trị threshold lấy từ
`validation.metrics.json`, không phải mặc định tùy ý.

## 19. Những lệnh không nên dùng

Không dùng nhánh NPZ legacy cho toàn bộ dataset:

```powershell
python -m cli.prepare_cert
```

Lệnh này chỉ phù hợp shard nhỏ/smoke và có thể tiêu tốn RAM lớn nếu áp dụng cho toàn
bộ `http.csv`.

Không:

- Copy thêm dataset vào repository.
- Đưa answer key vào Feature/Sequence.
- Fit scaler hoặc reference trên Validation/Test.
- Chọn threshold bằng Test label.
- Sửa trực tiếp manifest/checksum.
- Xóa SQLite store chỉ vì một epoch train bị dừng.

## 20. Checklist trước khi công bố kết quả

- [ ] ML tests và backend tests pass.
- [ ] Dataset là CERT R4.2 đầy đủ, không capped.
- [ ] Store đi đến ngày Test cuối `2011-05-17`.
- [ ] Scaler, PC profiles và references đều fit Train-only.
- [ ] Feature/Sequence schema version đúng.
- [ ] Validation threshold chọn không dùng label.
- [ ] Test chỉ dùng threshold đã khóa.
- [ ] Universe gồm cả inactive employee-days.
- [ ] Báo cáo coverage và NO_SCORE rate.
- [ ] Có `experiment_report.json`.
- [ ] Không gọi smoke metric là kết quả nghiên cứu.
