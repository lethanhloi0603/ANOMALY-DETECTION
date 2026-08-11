# Safe-update v5 rollout

Safe-update v5 phải được triển khai theo chế độ fail-closed. Việc thu thập và
đánh giá candidate có thể chạy ở shadow mode, nhưng không được tạo Personal
`ReferenceProfile` mới chỉ vì framework v5 đang là config mặc định.

## Hai gate materialization

Một release chỉ được materialize khi đồng thời thỏa cả hai điều kiện:

1. Contract versioned đặt
   `safe_personalized_update.release.materialization_enabled=true`.
2. Runtime đặt `SAFE_UPDATE_MATERIALIZATION_ENABLED=true`.

Mặc định của cả hai gate là `false`. Khi gate đóng, processor vẫn có thể kiểm
tra quarantine/watermark và chuyển candidate hợp lệ sang `ACCEPTED`; các
candidate đó tiếp tục nằm ở `deferred`, không đổi active accumulator pointer và
không tạo `ReferenceRelease` hay child `ReferenceProfile`.

## Điều kiện bắt buộc trước khi mở gate

- Database production là PostgreSQL.
- Không còn candidate legacy ở trạng thái `CANDIDATE` hoặc `ACCEPTED` trong
  toàn bộ database, không chỉ organization đang gọi processor. Hai activation
  gate là global theo environment nên một organization còn legacy pending sẽ
  chặn materialization cho mọi organization. Candidate
  legacy phải được retire bằng một thao tác riêng có actor, reason và audit;
  processor không được tự retire hoặc diễn giải lại bằng policy v5.
- PostgreSQL integration test hai worker đã chạy thành công và chứng minh chỉ có
  một child reference cho mỗi `(accumulator_id, release_sequence)`.
- Contract late-alert/compromise đã được duyệt và triển khai, gồm cách chọn
  ancestor sạch, xử lý descendants và cập nhật active pointer.
- Train reference/framework/checkpoint đã được tái tạo và xác minh theo
  `framework.v5`.

Nếu legacy gate còn pending khi cả hai materialization gate được yêu cầu mở,
processor phải dừng trước mọi mutation. Không được khắc phục bằng cách sửa trực
tiếp trạng thái candidate trong database. Lỗi chỉ công khai
`global_legacy_pending=true` và số lượng pending của organization hiện tại;
không rò rỉ số lượng candidate của organization khác.

## Trình tự rollout

1. Deploy schema và code với hai gate đóng.
2. Chạy admission, watermark và quarantine ở shadow mode; theo dõi `pending`,
   `deferred`, `rejected` và `legacy_pending`.
3. Retire legacy bằng workflow có audit sau khi Product/Ops duyệt semantics.
4. Chạy PostgreSQL concurrency suite và poisoning/sensitivity evaluation.
5. Triển khai late-alert/compromise workflow.
6. Bật contract gate trong một framework release đã review.
7. Bật runtime gate theo từng environment và giám sát immutable release lineage.

Tắt runtime gate là thao tác dừng materialization khẩn cấp; nó không rollback
reference đã phát hành và không thay thế late-alert/compromise workflow.
