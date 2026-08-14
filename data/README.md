# Data

- `raw/`: dữ liệu nguồn chỉ đọc; không sửa hoặc ghi đè.
- `interim/`: canonical events sau normalize/deduplicate.
- `processed/`: feature128, sequence7 và cửa sổ NPZ đưa vào model.
- `artifacts/`: scaler, model checkpoint, reference, manifest và báo cáo.
- `runtime/`: database/log local của backend.
- `evaluation/`: label và schema đánh giá tách biệt khỏi train/inference.

Các thư mục dữ liệu lớn được Git ignore. Mỗi artifact production phải có version,
checksum, khoảng thời gian fit và split nguồn để ngăn data leakage.

## Dataset CERT local

`raw/cert4.2` là Windows directory junction trỏ đến
`C:\Users\admin\Downloads\cert4.2`. File chỉ tồn tại tại thư mục nguồn; junction
không tạo thêm bản sao nên không làm tăng dung lượng dataset.

Pipeline feature/sequence chỉ được đọc các nguồn hành vi như `logon.csv`,
`device.csv`, `file.csv`, `http.csv`, `email.csv` và LDAP. Psychometric/OCEAN chỉ được dùng
trong ablation có schema và artifact tách riêng, không thuộc primary experiment.
`answers/` cùng `insiders.csv` là ground truth phục vụ evaluation, tuyệt đối
không được đưa vào preprocessing, train, reference hoặc model input.

`artifacts/experiment_v1` và `evaluation/experiment_v1` là baseline `framework.v5` và
không được ghi đè. Primary rerun theo Option 2 ghi vào `experiment_v2` với
`backend/config/framework.v6.json`; SQLite store trong `processed/` được tái sử dụng.
