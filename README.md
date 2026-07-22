# CERT R4.2 Multi-view Anomaly Detection

Framework phát hiện insider-threat anomaly theo đơn vị `user-day`, kết hợp
count/statistical view và raw token/source/time-gap sequence trong cửa sổ 30
calendar days.

Tài liệu nên đọc:

- [FRAMEWORK_FLOW.md](FRAMEWORK_FLOW.md): kiến trúc và luồng framework sau refactor.
- [ml/README_ML.md](ml/README_ML.md): câu lệnh prepare, train, evaluate và predict.
- [README_TUTORIAL.md](README_TUTORIAL.md): cài đặt backend/frontend/ML từ đầu.
- [README_SYSTEM.md](README_SYSTEM.md): mô tả chi tiết các thành phần hệ thống.

Thay đổi tài nguyên quan trọng: global training dùng lazy calendar windows nên
không còn tạo toàn bộ tensor window trong RAM. Prepare mặc định hỗ trợ output
`compact` để tránh ghi các CSV trung gian lớn.
