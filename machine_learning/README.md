# Machine learning

Thư mục này chỉ chứa code và cấu hình chạy ML:

```text
config/                 hợp đồng 128 feature, 7 token và kiến trúc model
insider_ml/
  contracts.py          shape/dtype model-ready
  dataset.py            đọc NPZ đã chuẩn bị
  model.py              TCN–Transformer Autoencoder
  temporal.py           time_sin/time_cos và baseline thời gian Person→Role→Global
  training.py           loss và một epoch train
  inference.py          reconstruction score và reference statistics
  evaluation/           metric nghiên cứu
cli/                    entrypoint chạy batch
tests/                  test contract/model/metric
```

Dữ liệu không nằm ở đây. Raw log ở `../data/raw`, tensor đã chuẩn bị ở
`../data/processed`, checkpoint và reference ở `../data/artifacts`.

Model nhận đồng thời feature 128 chiều và chuỗi event. TCN mã hóa quan hệ cục bộ,
Transformer học quan hệ dài hạn giữa các ngày, decoder tái tạo cả feature và token.
Feature error và Sequence error là hai anomaly score thô độc lập. Mỗi nhánh chọn và
calibrate bằng reference Person → Role → Global riêng; chỉ các percentile đã calibrate
mới được fusion.

Sequence contract `sequence7.v4` dùng `calendar_contexts` (`WEEKDAY/WEEKEND`) và hai kênh
liên tục `time_sin/time_cos`. Không có nhãn `WORK/AFTER`; thời gian bất thường được đo tương
đối với baseline quá khứ, không dựa trên một khoảng giờ làm việc tự đặt.

Chạy test từ thư mục này:

```powershell
..\.venv\Scripts\python.exe -m pytest
```

Train từ tensor đã chuẩn bị:

```powershell
..\.venv\Scripts\python.exe -m cli.train `
  --input ..\data\processed\train.npz `
  --output ..\data\artifacts\tcn_transformer_ae.v4.pt
```

Lệnh train ghi checkpoint và manifest checksum cạnh nhau. Không train trực tiếp từ
raw log; bước canonicalize/feature/sequence phải hoàn tất và khóa split trước.

Package không import FastAPI hoặc ORM của backend. Hai phần chỉ trao đổi qua hợp
đồng có version và artifact bất biến.

## Luồng chạy được từ raw CERT

Smoke test dùng dữ liệu CERT thật, có giới hạn rõ ràng và không sao chép thư mục raw:

```powershell
Set-Location .\machine_learning
..\.venv\Scripts\python.exe -m cli.run_framework
```

Lệnh này chạy liên tục `raw CSV + LDAP -> disk-backed Feature128/Sequence7 store ->
train 1 epoch -> last-day reconstruction score -> frozen Train reference check`. Kết quả nằm trong
`data/processed/smoke` và `data/artifacts/smoke`.

Có thể chạy riêng từng bước:

```powershell
..\.venv\Scripts\python.exe -m cli.prepare_cert --help
..\.venv\Scripts\python.exe -m cli.train --help
..\.venv\Scripts\python.exe -m cli.score --help
..\.venv\Scripts\python.exe -m cli.prepare_evaluation --help
```

`prepare_cert` chỉ giữ metadata cần thiết; trường `content` của file/http/email không được đưa
vào event nội bộ hay artifact. Timestamp được parse như local wall-clock CERT. Mẫu model luôn là
`[D-29,D]`; inference chỉ lấy reconstruction error tại ngày cuối `D`.

`prepare_cert` là materializer NPZ legacy dành cho smoke/user shard và vẫn gom shard trong RAM.
Không dùng lệnh đó cho toàn bộ HTTP 15 GB; thực nghiệm đầy đủ dùng `build_store`/`run_experiment`
ở phần dưới.

## Thực nghiệm đầy đủ trên máy 8 GB RAM

Pipeline chính thức không dùng NPZ lặp lại toàn bộ cửa sổ. `build_store` đọc từng raw source,
chỉ chọn metadata, giữ tối đa một ngày trong RAM rồi ghi aggregate nén vào SQLite. Train và
inference dựng cửa sổ `[D-29,D]` theo từng user trực tiếp từ store.

Chạy toàn bộ đến Test metric:

```powershell
Set-Location .\machine_learning
..\.venv\Scripts\python.exe -m cli.run_experiment `
  --batch-size 4 `
  --epochs 20 `
  --device cpu
```

Các stage được chạy theo thứ tự:

```text
raw CERT -> disk-backed user-day store
-> Train scaler/PC/time artifacts
-> train TCN-Transformer AE
-> frozen Train Person/Role/Global references
-> Validation predictions -> threshold q=0.995
-> Test predictions -> metrics với threshold đã khóa
```

Store checkpoint theo từng source. Checkpoint model được ghi sau mỗi epoch. Chạy lại cùng lệnh
sẽ resume các stage hợp lệ; checksum khác sẽ bị từ chối thay vì trộn artifact cũ và mới.

Smoke riêng cho low-RAM store:

```powershell
..\.venv\Scripts\python.exe -m cli.build_store `
  --store ..\data\processed\smoke_stream.sqlite `
  --max-rows-per-source 10000 `
  --end-day 2010-01-03

..\.venv\Scripts\python.exe -m cli.train `
  --store ..\data\processed\smoke_stream.sqlite `
  --output ..\data\artifacts\smoke\stream_model.pt `
  --epochs 1 --batch-size 4 --max-samples 8
```
