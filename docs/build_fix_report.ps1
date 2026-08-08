param(
    [string]$OutputPath = (Join-Path $PSScriptRoot "Bao_cao_sua_code_nhom_2_3_2026-08-03.docx"),
    [string]$PdfPath = (Join-Path $PSScriptRoot "Bao_cao_sua_code_nhom_2_3_2026-08-03.pdf")
)

$ErrorActionPreference = "Stop"
$word = $null
$document = $null

function Add-Paragraph {
    param([string]$Text, [string]$Style = "Normal")
    $paragraph = $document.Content.Paragraphs.Add()
    $paragraph.Range.Text = $Text
    $paragraph.Range.Style = $Style
    $paragraph.Range.InsertParagraphAfter()
}

function Add-Bullet {
    param([string]$Text)
    $paragraph = $document.Content.Paragraphs.Add()
    $paragraph.Range.Text = $Text
    $paragraph.Range.Style = "Normal"
    $paragraph.Range.ListFormat.ApplyBulletDefault()
    $paragraph.Range.InsertParagraphAfter()
}

function Add-Table {
    param([string[]]$Headers, [object[][]]$Rows, [int[]]$Widths)
    $range = $document.Content
    $range.Collapse(0)
    $table = $document.Tables.Add($range, $Rows.Count + 1, $Headers.Count)
    $table.Style = "Table Grid"
    $table.AllowAutoFit = $false
    for ($column = 1; $column -le $Headers.Count; $column++) {
        $table.Cell(1, $column).Range.Text = $Headers[$column - 1]
        $table.Cell(1, $column).Range.Bold = $true
        $table.Cell(1, $column).Shading.BackgroundPatternColor = 15007717
        $table.Columns.Item($column).Width = $word.CentimetersToPoints($Widths[$column - 1])
    }
    for ($row = 0; $row -lt $Rows.Count; $row++) {
        for ($column = 0; $column -lt $Headers.Count; $column++) {
            $table.Cell($row + 2, $column + 1).Range.Text = [string]$Rows[$row][$column]
        }
    }
    $table.Range.Font.Name = "Calibri"
    $table.Range.Font.Size = 9
    $table.Range.ParagraphFormat.SpaceAfter = 3
    $after = $document.Content.Paragraphs.Add()
    $after.Range.InsertParagraphAfter()
}

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Add()

    $section = $document.Sections.Item(1)
    $section.PageSetup.PaperSize = 2
    $section.PageSetup.TopMargin = $word.InchesToPoints(1)
    $section.PageSetup.BottomMargin = $word.InchesToPoints(1)
    $section.PageSetup.LeftMargin = $word.InchesToPoints(1)
    $section.PageSetup.RightMargin = $word.InchesToPoints(1)
    $section.PageSetup.HeaderDistance = $word.InchesToPoints(0.492)
    $section.PageSetup.FooterDistance = $word.InchesToPoints(0.492)

    $normal = $document.Styles.Item("Normal")
    $normal.Font.Name = "Calibri"
    $normal.Font.Size = 11
    $normal.ParagraphFormat.SpaceAfter = 6
    $normal.ParagraphFormat.LineSpacingRule = 5
    $normal.ParagraphFormat.LineSpacing = 13.75

    foreach ($name in @("Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3")) {
        $document.Styles.Item($name).Font.Name = "Calibri"
    }
    $document.Styles.Item("Title").Font.Size = 24
    $document.Styles.Item("Title").Font.Color = 10040064
    $document.Styles.Item("Heading 1").Font.Size = 16
    $document.Styles.Item("Heading 1").Font.Color = 10040064
    $document.Styles.Item("Heading 2").Font.Size = 13
    $document.Styles.Item("Heading 2").Font.Color = 10040064
    $document.Styles.Item("Heading 3").Font.Size = 12
    $document.Styles.Item("Heading 3").Font.Color = 6362112

    $header = $section.Headers.Item(1).Range
    $header.Text = "ANOMALY DETECTION  |  ENGINEERING CHANGE RECORD"
    $header.Font.Name = "Calibri"
    $header.Font.Size = 8
    $header.Font.Color = 8421504
    $footer = $section.Footers.Item(1).Range
    $footer.ParagraphFormat.Alignment = 2
    $footer.Text = "Internal technical record  •  03/08/2026  •  "
    $footer.Collapse(0)
    [void]$footer.Fields.Add($footer, 33)

    Add-Paragraph "BÁO CÁO SỬA CODE — NHÓM 2 & NHÓM 3" "Title"
    Add-Paragraph "Anomaly Detection Framework  |  Change record & recheck guide" "Subtitle"
    Add-Paragraph "Ngày lập: 03/08/2026     Phạm vi: backend API, database contract, scoring/training artifacts, kiểm thử hồi quy"
    Add-Paragraph "KẾT LUẬN NHANH" "Heading 1"
    Add-Paragraph "Các vấn đề có thể tự xử lý thuộc nhóm 2–3 đã được triển khai vào code. Những thay đổi tập trung vào tính nhất quán contract, chặn input vượt giới hạn runtime, chống hỏng artifact khi tiến trình bị ngắt, giảm RAM khi score, tải checkpoint an toàn hơn, khóa đồng thời khi phát hành assessment và kiểm tra evidence theo feature catalog."
    Add-Paragraph "Trạng thái xác minh" "Heading 2"
    Add-Bullet "Đã recheck tĩnh phạm vi file sửa và bổ sung test hồi quy."
    Add-Bullet "Chưa chạy được pytest trong phiên này vì workspace không có .venv và Python hệ thống không khả dụng. Vì vậy báo cáo không tuyên bố test đã pass."
    Add-Bullet "Các quyết định kiến trúc lớn (artifact reference v2, ledger/idempotency đầy đủ, chính sách personal/global baseline) không bị tự ý thay đổi."

    Add-Paragraph "1. DANH MỤC THAY ĐỔI" "Heading 1"
    Add-Table @("Hạng mục", "Mức tác động", "Trạng thái") @(
        @("Contract độ dài DB ↔ API", "Vừa", "Đã sửa + migration"),
        @("Giới hạn batch theo cấu hình runtime", "Vừa", "Đã sửa + test"),
        @("Request ID không an toàn", "Vừa", "Đã sửa + test"),
        @("Artifact ghi dở dang", "Vừa", "Đã sửa JSON/CSV/NPZ"),
        @("Score giữ toàn bộ rows trong RAM", "Vừa", "Đã stream phần output"),
        @("Checkpoint deserialization", "Vừa", "Đã bật weights_only"),
        @("Race khi phát hành assessment", "Vừa/Cao", "Đã khóa Postgres; SQLite giới hạn"),
        @("Evidence không khớp catalog", "Vừa", "Đã validate"),
        @("Evidence payload không giới hạn", "Thấp/Vừa", "Đã đặt trần")
    ) @(6, 4, 6)

    Add-Paragraph "2. CHI TIẾT KỸ THUẬT" "Heading 1"
    Add-Paragraph "2.1 Database contract và migration" "Heading 2"
    Add-Paragraph "Đã đồng bộ độ dài các cột original_id, pc, action, model_version, config_version, catalog_version và assignee với giới hạn request schema. Migration 0002 dùng batch_alter_table để chạy được trên SQLite và PostgreSQL. Downgrade kiểm tra dữ liệu trước khi thu hẹp cột nhằm tránh truncate âm thầm."
    Add-Paragraph "File: backend/app/models.py; backend/migrations/versions/0002_expand_api_contract_lengths.py"

    Add-Paragraph "2.2 Bảo vệ API input" "Heading 2"
    Add-Paragraph "POST /events giờ áp dụng min(MAX_BATCH_EVENTS, 1000) thay vì chỉ dựa vào giới hạn schema cố định. X-Request-ID chỉ nhận 1–160 ký tự thuộc [A-Za-z0-9_.:@-]; request sai trả INVALID_REQUEST_ID và không phản chiếu giá trị nguy hiểm vào response header."
    Add-Paragraph "Evidence JSON bị giới hạn 64 KiB; top_features và top_transitions tối đa 20 phần tử không rỗng. Feature evidence phải tồn tại trong feature catalog đang active."

    Add-Paragraph "2.3 Artifact an toàn khi tiến trình bị ngắt" "Heading 2"
    Add-Paragraph "Thêm helper atomic_write_json và atomic_write_csv: ghi file tạm cùng thư mục, flush/fsync rồi replace. Đã áp dụng cho score, train manifest, evaluation, experiment, CERT preparation, stream-store manifest và smoke report. NPZ prepared windows cũng được chuyển sang temp + fsync + replace."
    Add-Paragraph "File: machine_learning/insider_ml/artifacts.py và các CLI liên quan."

    Add-Paragraph "2.4 Bộ nhớ và checkpoint" "Heading 2"
    Add-Paragraph "Đường score từ SQLite store được chuyển thành iterator; calibrated rows và CSV output được xử lý tuần tự, không giữ toàn bộ prediction rows trong RAM. Khi fit reference v1, raw rows vẫn phải materialize vì cấu trúc artifact hiện tại cần toàn bộ phân phối — đây là giới hạn còn lại, không phải đã sửa hoàn toàn."
    Add-Paragraph "torch.load tại train resume và score đã dùng weights_only=True để giảm bề mặt deserialization ngoài ý muốn."

    Add-Paragraph "2.5 Đồng thời khi phát hành assessment" "Heading 2"
    Add-Paragraph "Trên PostgreSQL, transaction lấy pg_advisory_xact_lock theo khóa ổn định org/user/day/model/config trước khi tìm và tạo assessment. Cách này đóng race phổ biến giữa hai request đồng thời trong cùng release key. SQLite giữ no-op vì chỉ dùng dev/test và cơ chế writer serialization khác PostgreSQL."

    Add-Paragraph "3. TEST ĐÃ BỔ SUNG" "Heading 1"
    Add-Table @("Test", "Mục tiêu") @(
        @("test_request_id_rejects_unsafe_or_oversized_values", "Không phản chiếu request ID chứa CR/LF hoặc quá dài"),
        @("test_runtime_event_batch_limit_is_enforced", "MAX_BATCH_EVENTS runtime thực sự có hiệu lực"),
        @("test_atomic_json_replaces_destination", "JSON đích chỉ xuất hiện sau replace hoàn chỉnh"),
        @("test_atomic_csv_writes_complete_rows", "CSV hoàn chỉnh và không để lại temp file")
    ) @(7, 9)
    Add-Paragraph "Lệnh recheck đề xuất sau khi tạo môi trường Python:" "Heading 2"
    Add-Paragraph "Backend:  python -m pytest backend/tests -q"
    Add-Paragraph "ML:       python -m pytest machine_learning/tests -q"
    Add-Paragraph "Migration: alembic -c backend/alembic.ini upgrade head"

    Add-Paragraph "4. NHỮNG VIỆC CHƯA NÊN TỰ FIX" "Heading 1"
    Add-Table @("Vấn đề", "Vì sao cần bàn nhóm", "Đề xuất") @(
        @("Reference artifact v2 / streaming quantile", "Đổi format, checksum và khả năng tái lập model", "Lập ADR rồi benchmark sai số quantile"),
        @("Personal vs Global baseline policy", "Ảnh hưởng semantics score và ngưỡng cảnh báo", "Chốt fallback, cold-start, role epoch"),
        @("Assessment ledger/idempotency đầy đủ", "Đổi data model và lifecycle phát hành", "Thiết kế release key + unique constraint + retry"),
        @("SQLite production concurrency", "Advisory lock chỉ có ở PostgreSQL", "Không dùng SQLite production hoặc thêm lock strategy riêng")
    ) @(5, 6, 5)

    Add-Paragraph "5. CHECKLIST RECHECK SAU NÀY" "Heading 1"
    Add-Bullet "Chạy toàn bộ backend và ML tests trong môi trường dependency khóa phiên bản."
    Add-Bullet "Chạy migration trên bản sao dữ liệu production; kiểm tra downgrade guard."
    Add-Bullet "Fault-injection: kill process giữa lúc ghi CSV/JSON/NPZ và xác nhận file cũ còn nguyên."
    Add-Bullet "Load test hai request assessment cùng release key trên PostgreSQL."
    Add-Bullet "Benchmark peak RAM của score store; tách riêng trường hợp --fit-reference-out."
    Add-Bullet "Xác nhận mọi feature evidence gửi từ ML đều thuộc feature catalog active."

    Add-Paragraph "6. FILE ĐÃ THAY ĐỔI" "Heading 1"
    Add-Paragraph "Backend: app/models.py, app/api.py, app/web.py, app/schemas.py, app/services.py, migration 0002, tests/test_api.py."
    Add-Paragraph "Machine learning: insider_ml/artifacts.py, insider_ml/cert_data.py, insider_ml/stream_store.py, cli/score.py, cli/train.py, cli/evaluate_metrics.py, cli/prepare_evaluation.py, cli/prepare_cert.py, cli/run_experiment.py, cli/run_framework.py, tests/test_artifacts.py."

    Add-Paragraph "Ghi chú phạm vi" "Heading 2"
    Add-Paragraph "Báo cáo này là change record cho nhóm 2–3. Nó không xác nhận các vấn đề nhóm 1 đã được giải quyết và không thay thế ADR cho các thay đổi nghiệp vụ/kiến trúc lớn."

    $document.Repaginate()
    $document.SaveAs2($OutputPath, 16)
    $document.ExportAsFixedFormat($PdfPath, 17)
    $pageCount = $document.ComputeStatistics(2)
    $document.Close($false)
    $word.Quit()
    Write-Output "DOCX=$OutputPath"
    Write-Output "PDF=$PdfPath"
    Write-Output "PAGES=$pageCount"
}
finally {
    if ($null -ne $document) { try { $document.Close($false) } catch {} }
    if ($null -ne $word) { try { $word.Quit() } catch {} }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($document) 2>$null | Out-Null
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) 2>$null | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
