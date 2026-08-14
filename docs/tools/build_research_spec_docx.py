from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\admin\Downloads\[NCKH] NỘI DUNG (3).docx")
OUTPUT = ROOT / "Dac_ta_NCKH_TCN_Transformer_Person_Role_Global.docx"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def run(text: str, *, bold: bool = False, italic: bool = False, size: int | None = None,
        color: str | None = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if italic:
        props.append("<w:i/>")
    if size:
        props.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    if color:
        props.append(f'<w:color w:val="{color}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def para(text: str = "", *, style: str | None = None, bold: bool = False,
         italic: bool = False, align: str | None = None, before: int = 0,
         after: int = 120, page_break_before: bool = False,
         keep_next: bool = False, color: str | None = None,
         size: int | None = None) -> str:
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    ppr.append(f'<w:spacing w:before="{before}" w:after="{after}" w:line="300" w:lineRule="auto"/>')
    if page_break_before:
        ppr.append("<w:pageBreakBefore/>")
    if keep_next:
        ppr.append("<w:keepNext/>")
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{run(text, bold=bold, italic=italic, size=size, color=color)}</w:p>"


def bullet(text: str, *, level: int = 0) -> str:
    left = 720 + level * 360
    return (
        f'<w:p><w:pPr><w:pStyle w:val="ListParagraph"/>'
        f'<w:numPr><w:ilvl w:val="{level}"/><w:numId w:val="1"/></w:numPr>'
        f'<w:ind w:left="{left}" w:hanging="360"/>'
        '<w:spacing w:after="80" w:line="290" w:lineRule="auto"/>'
        f'</w:pPr>{run(text)}</w:p>'
    )


def heading(text: str, level: int = 1) -> str:
    return para(
        text,
        style=f"Heading{level}",
        before={1: 320, 2: 240, 3: 160}[level],
        after={1: 160, 2: 120, 3: 80}[level],
        keep_next=True,
    )


def table(headers: list[str], rows: list[list[str]], widths: list[int]) -> str:
    total = sum(widths)
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [
        '<w:tbl><w:tblPr>',
        f'<w:tblW w:w="{total}" w:type="dxa"/>',
        '<w:tblInd w:w="120" w:type="dxa"/>',
        '<w:tblLayout w:type="fixed"/>',
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:left w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:right w:val="single" w:sz="4" w:color="AAB7C4"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="D9E0E7"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="D9E0E7"/></w:tblBorders>',
        '<w:tblCellMar><w:top w:w="90" w:type="dxa"/><w:left w:w="120" w:type="dxa"/>'
        '<w:bottom w:w="90" w:type="dxa"/><w:right w:w="120" w:type="dxa"/></w:tblCellMar>',
        '</w:tblPr>',
        f"<w:tblGrid>{grid}</w:tblGrid>",
    ]
    all_rows = [headers, *rows]
    for ridx, row in enumerate(all_rows):
        out.append("<w:tr>")
        for cidx, text in enumerate(row):
            fill = '<w:shd w:fill="E8EEF5"/>' if ridx == 0 else ""
            out.append(
                f'<w:tc><w:tcPr><w:tcW w:w="{widths[cidx]}" w:type="dxa"/>{fill}'
                '<w:vAlign w:val="center"/></w:tcPr>'
                f'{para(text, bold=ridx == 0, after=40)}</w:tc>'
            )
        out.append("</w:tr>")
    out.append("</w:tbl>")
    out.append(para("", after=80))
    return "".join(out)


def callout(label: str, text: str) -> str:
    return table([label], [[text]], [9360])


def build_body() -> str:
    x: list[str] = []
    x += [
        para("ĐẶC TẢ NGHIÊN CỨU VÀ TRIỂN KHAI", bold=True, align="center",
             size=44, color="1F4D78", after=120),
        para("Phát hiện bất thường hành vi nội bộ bằng TCN–Transformer Autoencoder",
             bold=True, align="center", size=34, color="2E74B5", after=80),
        para("Kết hợp đặc trưng người dùng và chuỗi hành vi với cơ chế tham chiếu "
             "Person → Role → Global", align="center", italic=True, size=26, after=300),
        table(
            ["Thuộc tính", "Giá trị"],
            [
                ["Loại tài liệu", "Research specification kiêm đặc tả nghiệp vụ/kỹ thuật cho đội phát triển"],
                ["Đơn vị phân tích", "Một user-day: (user_id, ngày D)"],
                ["Mô hình chính", "TCN–Transformer Autoencoder không giám sát trên cửa sổ 30 ngày"],
                ["Cơ chế tham chiếu", "Đúng ba cấp: Person → Role → Global"],
                ["Mục tiêu đầu ra", "Risk score, alert, bằng chứng giải thích và trạng thái fallback"],
                ["Trạng thái", "Bản đặc tả để duyệt trước khi triển khai"],
            ],
            [2100, 7260],
        ),
        callout("QUYẾT ĐỊNH ĐÃ KHÓA",
                "Giữ mô hình TCN–Transformer Autoencoder làm mô hình nghiên cứu chính. "
                "Giữ đúng ba cấp Person, Role và Global. Feature và Sequence được xử lý "
                "độc lập đến bước calibration; fallback chỉ xảy ra khi thiếu support."),
        para("Mục lục nội dung", style="Heading1", page_break_before=True),
    ]
    for item in [
        "1. Tóm tắt đề tài và vấn đề nghiên cứu",
        "2. Mục tiêu, câu hỏi và giả thuyết nghiên cứu",
        "3. Phạm vi, actor và nguyên tắc nghiệp vụ",
        "4. Hợp đồng dữ liệu và phòng chống leakage",
        "5. Feature branch và Sequence branch",
        "6. TCN–Transformer Autoencoder",
        "7. Readiness và fallback Person → Role → Global",
        "8. Calibration, fusion và alert",
        "9. Safe personalized update",
        "10. Pipeline triển khai cho dev",
        "11. Artifact, API và persistence",
        "12. Thiết kế thực nghiệm và chỉ số đánh giá",
        "13. Test matrix và acceptance criteria",
        "14. Backlog, Definition of Done và rủi ro",
    ]:
        x.append(bullet(item))

    x += [
        heading("1. Tóm tắt đề tài và vấn đề nghiên cứu"),
        para("Đề tài xây dựng hệ thống phát hiện bất thường hành vi người dùng nội bộ từ năm nguồn "
             "log CERT: logon, device, file, HTTP và email, kết hợp ngữ cảnh vai trò từ LDAP. "
             "Hệ thống không học trực tiếp nhãn kịch bản mà học hành vi thông thường bằng "
             "TCN–Transformer Autoencoder. Điểm bất thường được tính cho từng user-day."),
        para("Khó khăn cốt lõi là dữ liệu lịch sử không đồng đều. Người dùng mới hoặc vừa đổi vai "
             "trò không có đủ lịch sử cá nhân; role nhỏ có thể không đủ peer. Vì vậy mỗi nhánh "
             "Feature và Sequence chọn đúng một reference distribution theo thứ tự Person, Role, "
             "Global. Global là fallback cuối cùng của toàn công ty."),
        heading("1.1. Phát biểu bài toán", 2),
        para("Cho cửa sổ 30 ngày W(u,D) của user u kết thúc tại ngày D, hệ thống phải sinh hai raw "
             "anomaly score E_F và E_S, lựa chọn tầng tham chiếu hợp lệ L_F và L_S, calibration "
             "thành q_F và q_S trong [0,1], sau đó fusion thành risk. q càng cao càng bất thường."),
        heading("1.2. Đóng góp dự kiến", 2),
    ]
    for s in [
        "Kiến trúc multimodal kết hợp Feature128 và Sequence7 trong một mô hình TCN–Transformer Autoencoder.",
        "Cơ chế readiness và hierarchical backoff độc lập cho từng branch, duy trì coverage khi Person hoặc Role thiếu dữ liệu.",
        "Cơ chế calibration theo đúng branch và level đã chọn, tránh cộng lặp ba score Person/Role/Global.",
        "Safe-update có quarantine nhằm giảm baseline poisoning khi personal profile được cập nhật online.",
        "Bộ thực nghiệm ablation tách giá trị của feature, sequence, fusion, hierarchy và deep model.",
    ]:
        x.append(bullet(s))

    x += [
        heading("2. Mục tiêu, câu hỏi và giả thuyết nghiên cứu"),
        table(
            ["Mã", "Câu hỏi nghiên cứu", "Giả thuyết kiểm chứng"],
            [
                ["RQ1", "Feature + Sequence có tốt hơn từng branch riêng?", "Fusion tăng AUPRC/Recall@K hoặc giảm false alert tại cùng budget."],
                ["RQ2", "Ba cấp P→R→G có tốt hơn Global-only?", "Hierarchy tăng coverage và giảm false positive mà không làm giảm recall đáng kể."],
                ["RQ3", "TCN–Transformer AE có tốt hơn robust+n-gram?", "Deep model học được phụ thuộc chuỗi và thời gian vượt baseline trên Test đã khóa."],
                ["RQ4", "Safe update có giúp thích nghi mà không poisoning?", "Quarantine tốt hơn immediate update và gần frozen-person về độ ổn định."],
            ],
            [900, 3900, 4560],
        ),
        heading("2.1. Tiêu chí thành công", 2),
    ]
    for s in [
        "Pipeline chạy tái lập từ raw data đến risk/alert bằng manifest và checksum.",
        "Không có label/scenario trong core data, feature, sequence, model input hoặc threshold selection trên Test.",
        "Mỗi alert giải thích được level, support, fallback reason, top feature và top transition.",
        "Kết quả báo cáo AUPRC, recall, alert/day, false alert/1.000 user-day, time-to-detect và confidence interval.",
        "Mọi hyperparameter được chọn trên Validation và khóa trước Test.",
    ]:
        x.append(bullet(s))

    x += [
        heading("3. Phạm vi, actor và nguyên tắc nghiệp vụ"),
        table(
            ["Actor", "Trách nhiệm", "Không được phép"],
            [
                ["Researcher", "Khóa split, feature/token schema, thực nghiệm và tiêu chí đánh giá", "Sửa rule sau khi xem Test để cải thiện kết quả"],
                ["Data engineer", "Ingest, normalize, tạo feature/sequence và checkpoint", "Đưa label/scenario vào core artifact"],
                ["ML engineer", "Train/freeze model, fit calibrator, sinh score", "Tự khai support không truy xuất từ artifact"],
                ["Backend developer", "Enforce readiness, fallback, fusion, audit và idempotency", "Cho request ghi đè weight/threshold đang active"],
                ["Analyst", "Xem, acknowledge, điều tra và đóng alert", "Sửa risk/score nguồn"],
            ],
            [1500, 4300, 3560],
        ),
        heading("3.1. Nguyên tắc bất biến", 2),
    ]
    for s in [
        "Đơn vị chấm điểm là user-day; cửa sổ mô hình gồm 30 ngày kết thúc tại D.",
        "Năm nguồn log được chuẩn hóa rồi append dọc vào canonical event; không join ngang event-level.",
        "Feature và Sequence có readiness, level, score và evidence riêng.",
        "Fallback chỉ do thiếu support; score cao hoặc transition mới là anomaly, không phải lý do fallback.",
        "Mọi profile/support dùng khi chấm D chỉ chứa dữ liệu trước D.",
        "Role/Global được fit Train-only và frozen trong Validation/Test.",
        "Scenario labels chỉ dùng trong evaluation process vật lý tách biệt.",
    ]:
        x.append(bullet(s))

    x += [
        heading("4. Hợp đồng dữ liệu và phòng chống leakage"),
        table(
            ["Artifact", "Khóa", "Nội dung", "Invariant"],
            [
                ["CanonicalEvent", "org,event_uid", "source, timestamp, user, pc, action, object, context", "Không label; idempotent"],
                ["Feature128", "org,user,day,version", "Đúng 128 values + masks", "Không NaN/Inf; ratio [0,1]"],
                ["Sequence7", "org,user,day,version", "tokens, gap, pc/time context, event_uid", "Stable order; max 256"],
                ["RoleContext", "user,valid_from", "role effective-dated", "Không snapshot tương lai"],
                ["Reference", "branch,level,scope,version", "support, fit range, calibrator, checksum", "Versioned; frozen theo split"],
                ["Assessment", "user,day,model,config", "scores, levels, risk, reasons", "Immutable và idempotent"],
            ],
            [1550, 1900, 3410, 2500],
        ),
        heading("4.1. Chuẩn hóa thời gian và thứ tự", 2),
        para("Dev phải chốt timezone chuẩn, ranh giới ngày, giờ làm việc và weekend trong config có "
             "version. Event cùng timestamp được sắp theo (timestamp, source_rank, original_id). "
             "Thay đổi tie-order phải tăng sequence schema version."),
        heading("4.2. Missing day và mask", 2),
        para("Inactive day, ngày trước khi user xuất hiện và padding của cửa sổ 30 ngày là ba trạng "
             "thái khác nhau. Mô hình phải nhận day_mask và activity_mask; không biểu diễn cả ba "
             "bằng cùng vector 0 mà không có mask."),

        heading("5. Feature branch và Sequence branch"),
        heading("5.1. Feature branch", 2),
        para("Input X_F(u,D) có đúng 128 đặc trưng thuộc Logon 18, Device 12, File 24, HTTP 24, "
             "Email 26, Cross-source 16 và History 8. Count/duration dùng log1p; scaler chỉ fit "
             "Train. Ratio giữ [0,1]; minute-of-day dùng biểu diễn vòng tròn hoặc khoảng cách vòng."),
        heading("5.2. Sequence branch", 2),
        para("Vocabulary nghiệp vụ giữ đúng bảy token khách quan: LOGON, LOGOFF, "
             "DEVICE_CONNECT, DEVICE_DISCONNECT, FILE, HTTP và EMAIL. "
             "OWN/SHARED/FOREIGN, WEEKDAY/WEEKEND, time_sin/time_cos, "
             "gap bucket và action "
             "chi tiết là side channel. Sequence sort ổn định và truncate head-128 + tail-128."),
        callout("RULE SEQUENCE NO_SCORE",
                "Nếu user-day có seq_len < 2 thì không có transition đủ nghĩa. Sequence branch trả "
                "NO_SCORE ngay; không chuyển Person sang Role/Global. Fusion chỉ dùng Feature nếu có."),

        heading("6. TCN–Transformer Autoencoder"),
        table(
            ["Khối", "Đặc tả bắt buộc", "Output"],
            [
                ["Feature encoder", "MLP 128→128→64, mask-aware, dropout cấu hình", "z_F(D) ∈ R64"],
                ["Daily sequence encoder", "Token/context/gap embeddings + positional encoding; pooling/attention", "z_S(D) ∈ R64"],
                ["Day representation", "Concat z_F,z_S + availability/day masks", "z_day(D) ∈ R128"],
                ["TCN", "4 residual blocks; kernel 3; dilation 1/2/4/8; causal", "Temporal local representation"],
                ["Transformer", "2 layers; d_model 128; 4 heads; FF 256; causal/padding mask", "Long-range representation"],
                ["Decoders", "Feature reconstruction head và Sequence token/context reconstruction head", "X_hat và token logits"],
                ["Score heads", "Loss ngày cuối và evidence theo branch", "E_F, E_S"],
            ],
            [1600, 5500, 2260],
        ),
        heading("6.1. Loss function", 2),
        para("Loss tổng phải tách được theo branch: L = λF·L_feature + λS·L_sequence + "
             "λaux·L_next_token. Feature dùng masked Huber/MSE theo loại biến; Sequence dùng "
             "cross-entropy token và side-channel phù hợp. Không chỉ tái tạo embedding hợp nhất, "
             "vì khi đó không chứng minh được mô hình giữ thông tin thứ tự."),
        heading("6.2. Raw anomaly score", 2),
        para("E_F và E_S được tính riêng cho ngày D từ reconstruction error đã chuẩn hóa theo "
             "dimension khả dụng. Model chung không thay đổi theo Person/Role/Global. Ba cấp chỉ "
             "tham gia ở readiness và calibration sau raw score."),
        heading("6.3. Training protocol", 2),
    ]
    for s in [
        "Fit tokenizer, scaler, context mapping và model chỉ trên Train.",
        "Early stopping và model selection dùng Validation; Test chỉ inference một lần theo config đã khóa.",
        "Không dùng scenario labels trong loss, sampling hoặc representation learning.",
        "Lưu seed, code version, config checksum, data manifest, optimizer state và best checkpoint.",
        "Baseline robust Feature + smoothed n-gram Sequence là đối chứng bắt buộc, không thay mô hình chính.",
    ]:
        x.append(bullet(s))

    x += [
        heading("7. Readiness và fallback Person → Role → Global"),
        heading("7.1. Decision rule chung", 2),
        para("Với mỗi branch B ∈ {FEATURE, SEQUENCE}, backend lần lượt đánh giá eligibility của "
             "PERSON, ROLE và GLOBAL. Chọn level đầu tiên đạt toàn bộ điều kiện. Không cộng score "
             "của ba level. Nếu Global không đủ, branch trả NO_SCORE."),
        table(
            ["Branch/Level", "Điều kiện khởi tạo", "Kết quả khi thiếu"],
            [
                ["Feature/Person", "≥60 active days; span ≥90; ≥60 ngày trong role hiện tại; coverage ≥0,90; mỗi feature đã dùng có ≥40 quan sát; stale ≤30", "Thử Feature/Role"],
                ["Feature/Role", "Role known; peer users loại subject ≥15; peer user-days ≥300; recent support đủ", "Thử Feature/Global"],
                ["Feature/Global", "Users ≥200; user-days ≥10.000; coverage ≥0,90; Train-only", "Feature NO_SCORE"],
                ["Sequence/Person", "≥60 sequence-days; transitions ≥1.500; span ≥90; ≥60 ngày trong role; stale ≤30", "Thử Sequence/Role"],
                ["Sequence/Role", "Role known; peer users ≥15; sequence-days ≥300; transitions ≥10.000", "Thử Sequence/Global"],
                ["Sequence/Global", "Users ≥200; sequence-days ≥10.000; transitions ≥100.000; Train-only", "Sequence NO_SCORE"],
            ],
            [1900, 5400, 2060],
        ),
        para("Các số trên là default nghiên cứu, không phải chân lý nghiệp vụ. Owner là Research "
             "Lead; dev phải đưa vào config versioned và cung cấp sensitivity grid trên Validation."),
        heading("7.2. Role change", 2),
        para("Personal profile gắn với role epoch [valid_from, valid_to). Khi role đổi, profile cũ "
             "đóng và user warm-up lại ở role mới. Trong warm-up thử Role rồi Global. Không gộp "
             "role nhỏ sau khi xem Test."),
        heading("7.3. Reason codes tối thiểu", 2),
    ]
    for s in [
        "PERSON: SUPPORT_MISSING, DAYS_LOW, SPAN_LOW, ROLE_TENURE_LOW, COVERAGE_LOW, STALE.",
        "ROLE: ROLE_UNKNOWN, USERS_LOW, DAYS_LOW, TRANSITIONS_LOW, RECENT_LOW, COVERAGE_LOW.",
        "GLOBAL: USERS_LOW, DAYS_LOW, TRANSITIONS_LOW, NOT_TRAIN_FITTED.",
        "CURRENT DAY: SEQ_LEN_LOW.",
    ]:
        x.append(bullet(s))

    x += [
        heading("8. Calibration, fusion và alert"),
        heading("8.1. Calibration contract", 2),
        para("Với raw score E_B và level L_B đã chọn, calibrator C(B,L,scope,version) biến score "
             "thành q_B = C(E_B) trong [0,1], quy ước q càng cao càng bất thường. Mỗi calibrator "
             "phải lưu sample count, fitted_from, fitted_through, method, checksum và update date."),
        heading("8.2. Fusion", 2),
        table(
            ["Trạng thái", "Risk rule"],
            [
                ["F và S có score", "risk = wF·qF + wS·qS; wF+wS=1"],
                ["Chỉ Feature", "risk = qF; effective weight F=1"],
                ["Chỉ Sequence", "risk = qS; effective weight S=1"],
                ["Cả hai NO_SCORE", "risk=NULL; không tạo alert"],
            ],
            [2700, 6660],
        ),
        para("Risk là anomaly index, không được diễn giải là xác suất user là insider. Weight và "
             "alert threshold được chọn trên Validation theo alert budget và khóa trước Test."),
        heading("8.3. Alert policy", 2),
        para("Tạo alert khi risk ≥ threshold. Assessment và branch score bất biến; alert workflow "
             "có thể OPEN, ACKNOWLEDGED, IN_REVIEW, RESOLVED, DISMISSED hoặc FALSE_POSITIVE. "
             "Mọi thay đổi phải có actor, timestamp, request_id và audit event."),

        heading("9. Safe personalized update"),
        table(
            ["State", "Điều kiện", "Hành động"],
            [
                ["CANDIDATE", "User-day đã score bằng profile chưa chứa ngày D", "Đưa vào quarantine"],
                ["REJECTED", "Có alert trong [D,D+30]; ROLE/GLOBAL parent CDF ≥0,90; role epoch đổi", "Không cập nhật Person"],
                ["ACCEPTED", "Hết quarantine và không có reject reason", "Đủ điều kiện tạo profile version mới"],
                ["APPLIED", "Profile mới ghi thành công và checksum hợp lệ", "Đóng version cũ; audit before/after"],
            ],
            [1450, 4600, 3310],
        ),
        para("Safe-update dùng cửa sổ quarantine đóng [D,D+30]; ngày sớm nhất đủ điều kiện là "
             "D+31. Incremental release bị giới hạn tối đa 2% mỗi release, cách nhau ít nhất 7 "
             "ngày, và 10% trong cửa sổ rolling 30 ngày; bootstrap đủ support được miễn các cap "
             "influence này. Không update in-place. Dev phải tạo immutable Person profile version "
             "mới để rollback và tái lập. Thực nghiệm phải so frozen-person, immediate-update và "
             "safe-update."),

        heading("10. Pipeline triển khai cho dev"),
        table(
            ["Pha", "Đầu vào", "Đầu ra/DoD"],
            [
                ["P00 Audit", "5 CSV + LDAP", "Schema, row count, date range, checksum, memory estimate"],
                ["P01 Identity/context", "LDAP + logon Train", "Role epochs; PC ownership; config version"],
                ["P02 Canonicalization", "Raw rows", "Append-only canonical events; row conservation"],
                ["P03 Feature materialization", "Canonical events", "Feature128 + masks; unique user-day"],
                ["P04 Sequence materialization", "Canonical events", "Sequence7; stable order; truncate audit"],
                ["P05 Train model", "Train windows", "Frozen model artifact + manifest"],
                ["P06 Raw scoring", "30-day windows", "E_F/E_S + reconstruction evidence"],
                ["P07 References", "Past scores/support", "Versioned P/R/G calibrators"],
                ["P08 Decision", "Scores + readiness", "Levels, qF/qS, risk, alert"],
                ["P09 Evaluation", "Decision export + labels DB", "Metrics, CI, ablation report"],
            ],
            [1300, 3050, 5010],
        ),
        heading("10.1. Operational requirements", 2),
    ]
    for s in [
        "Streaming/chunk ingestion; không đọc toàn bộ http.csv vào RAM.",
        "Checkpoint theo source/month; retry không tạo event trùng.",
        "Atomic artifact write bằng temp + checksum + rename.",
        "Mỗi pipeline stage có progress, timeout, error manifest và khả năng resume.",
        "Không cấp evaluation credential cho API, ingestion worker hoặc scorer.",
    ]:
        x.append(bullet(s))

    x += [
        heading("11. Artifact, API và persistence"),
        table(
            ["Module/API", "Hành vi bắt buộc", "Kiểm soát"],
            [
                ["Ingestion", "Tạo job, checkpoint, canonical batch", "Idempotency key + manifest hash"],
                ["Feature/Sequence", "Upsert theo user-day-schema", "Exact schema, checksum, source evidence"],
                ["Model registry", "Đăng ký immutable model/config release", "Artifact checksum + approval"],
                ["Reference registry", "Tạo P/R/G calibrator profile", "Scope/effective date/frozen checks"],
                ["Scoring", "Server-side raw score/readiness/fusion", "Dedicated scorer identity"],
                ["Alerts", "List/read/workflow", "RBAC + immutable assessment"],
                ["Safe update", "Process candidate và tạo profile version", "Locking + idempotency + rollback"],
            ],
            [1850, 4700, 2810],
        ),
        heading("11.1. Database invariants", 2),
    ]
    for s in [
        "Không có hai role assignment overlap cho cùng user.",
        "Không có hai event cùng event_uid trong organization.",
        "Một assessment duy nhất cho user-day-model-config.",
        "SCORED bắt buộc có risk/threshold và ít nhất một branch score; NO_SCORE không được tạo alert.",
        "Reference dùng cho D có fitted_through < D.",
        "Safe-update candidate chỉ được APPLIED một lần.",
    ]:
        x.append(bullet(s))

    x += [
        heading("12. Thiết kế thực nghiệm và chỉ số đánh giá"),
        table(
            ["Run", "Thiết kế", "Mục đích"],
            [
                ["E0", "48 Feature, Global, robust baseline", "Baseline volume đơn giản"],
                ["E1", "128 Feature, Global", "Giá trị feature mở rộng"],
                ["E2", "Sequence-only, Global", "Giá trị thứ tự"],
                ["E3", "Feature + Sequence, Global", "Giá trị fusion"],
                ["E4", "Fusion, Role→Global", "Giá trị role context"],
                ["E5", "Fusion, Person→Role→Global", "Framework đầy đủ ba cấp"],
                ["E6", "E5 với frozen/immediate/safe update", "Poisoning và thích nghi"],
                ["E7", "TCN–Transformer AE so với robust+n-gram", "Giá trị deep model"],
            ],
            [900, 4200, 4260],
        ),
        heading("12.1. Split", 2),
        para("Train: 02/01/2010–31/05/2010. Validation: 01/06/2010–30/09/2010. Test: "
             "01/10/2010–17/05/2011. Không random split user-day. Test online phải ghi rõ là "
             "prequential; mọi update chỉ dùng ngày quá khứ và rule đã khóa."),
        heading("12.2. Metrics", 2),
    ]
    for s in [
        "Primary: AUPRC theo user-day.",
        "Threshold metrics: Precision, Recall, F1 tại cùng alert budget.",
        "Operations: false alerts/1.000 user-day, alerts/day, Recall@Top-K/day.",
        "Incident: detection rate, mean/median time-to-detect.",
        "Diagnostics: fallback rate P/R/G/NO_SCORE theo branch và coverage.",
        "Uncertainty: 95% confidence interval bằng user-block bootstrap; công bố denominator.",
    ]:
        x.append(bullet(s))

    x += [
        heading("13. Test matrix và acceptance criteria"),
        table(
            ["Mã", "Given/When/Then"],
            [
                ["AC01", "Given Feature đủ Person và Sequence chỉ đủ Role, when score, then F=PERSON, S=ROLE."],
                ["AC02", "Given Person thiếu, Role đủ, when select reference, then dùng đúng Role và không trộn Person."],
                ["AC03", "Given Role UNKNOWN, when fallback, then bỏ Role và thử Global."],
                ["AC04", "Given seq_len<2, when score, then Sequence=NO_SCORE dù Role/Global đủ."],
                ["AC05", "Given một branch NO_SCORE, when fusion, then branch còn lại có weight hiệu dụng 1."],
                ["AC06", "Given cả hai NO_SCORE, when decision, then risk=NULL và không tạo alert."],
                ["AC07", "Given support_as_of≥D, when scoring, then từ chối vì temporal leakage."],
                ["AC08", "Given retry cùng payload/model/config, then trả assessment cũ; payload khác trả conflict."],
                ["AC09", "Given candidate hết quarantine và hợp lệ, then tạo Person profile version mới và audit checksum."],
                ["AC10", "Given hai worker cùng apply candidate, then chỉ một lần thành công."],
                ["AC11", "Given config/model checksum thay đổi nhưng giữ version, then release bị từ chối."],
                ["AC12", "Given Test labels, then core/model pipeline không có quyền đọc."],
            ],
            [900, 8460],
        ),
        heading("14. Backlog, Definition of Done và rủi ro"),
        table(
            ["Ưu tiên", "Epic", "Điều kiện hoàn tất"],
            [
                ["P0", "Canonical data + label firewall", "Row conservation, idempotency, leakage tests"],
                ["P0", "Feature128 + Sequence7", "Exact schema, masks, deterministic checksum"],
                ["P0", "Readiness/fallback/fusion", "AC01–AC08 pass"],
                ["P1", "TCN–Transformer training/inference", "Frozen artifact, reproducible E_F/E_S"],
                ["P1", "Reference/calibration registry", "Versioned P/R/G profiles"],
                ["P1", "Evaluation E0–E7", "Locked Test report + CI"],
                ["P2", "Safe-update APPLY", "Immutable version, concurrency, rollback"],
                ["P2", "Analyst workflow/RBAC", "Permission matrix + audit"],
            ],
            [900, 3500, 4960],
        ),
        heading("14.1. Rủi ro nghiên cứu", 2),
    ]
    for s in [
        "Sequence 7 token có thể quá thô; bắt buộc đo giá trị side channel và E2/E3.",
        "Hand-crafted risk catalog có thể leakage kịch bản; phải version và ablation không catalog.",
        "Ngưỡng readiness tùy ý; phải sensitivity analysis trên Validation.",
        "Percentile fusion không phải probability; chỉ gọi anomaly index.",
        "Personal online update có thể poisoning; E6 là bắt buộc.",
        "CERT là dữ liệu tổng hợp; không khẳng định khả năng tổng quát cho doanh nghiệp thật nếu chưa external validation.",
    ]:
        x.append(bullet(s))
    x += [
        heading("14.2. Definition of Done cuối cùng", 2),
        para("Đề tài chỉ được xem là hoàn tất khi pipeline tái lập chạy từ raw logs đến alert; "
             "TCN–Transformer AE sinh được E_F/E_S; ba cấp fallback được enforce độc lập; "
             "E0–E7 chạy trên split thời gian đã khóa; Test không được dùng để tinh chỉnh; "
             "mọi kết quả có manifest, checksum, metrics, confidence interval và phân tích giới hạn."),
        callout("KẾT LUẬN PHÊ DUYỆT",
                "Kiến trúc được đề xuất giữ đúng TCN–Transformer Autoencoder và đúng ba cấp "
                "Person → Role → Global. Dev không được tự thay đổi số cấp, trộn score của ba "
                "reference hoặc dùng Test để chọn rule. Mọi ngưỡng số phải là config versioned "
                "và được xác nhận bằng Validation/sensitivity analysis."),
    ]
    return "".join(x)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    shutil.copy2(SOURCE, OUTPUT)
    body = build_body()
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{W}"><w:body>{body}'
        '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1080" w:right="1440" w:bottom="1080" w:left="1440" '
        'w:header="720" w:footer="720" w:gutter="0"/>'
        '<w:cols w:space="720"/><w:docGrid w:linePitch="360"/></w:sectPr>'
        '</w:body></w:document>'
    )
    temp = OUTPUT.with_suffix(".tmp.docx")
    with zipfile.ZipFile(OUTPUT, "r") as zin, zipfile.ZipFile(
        temp, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            if item.filename == "word/document.xml":
                zout.writestr(item, document_xml.encode("utf-8"))
            else:
                zout.writestr(item, zin.read(item.filename))
    temp.replace(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
