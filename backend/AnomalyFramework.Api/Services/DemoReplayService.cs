using System.Globalization;
using System.Text.Json;
using AnomalyFramework.Api.Data;
using AnomalyFramework.Api.Dtos;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace AnomalyFramework.Api.Services;

public class DemoReplayService
{
    private readonly RuntimePaths _paths;
    private readonly MlSettings _settings;
    private readonly DetectionService _detectionService;
    private readonly AppDbContext _db;

    public DemoReplayService(
        RuntimePaths paths,
        IOptions<MlSettings> options,
        DetectionService detectionService,
        AppDbContext db)
    {
        _paths = paths;
        _settings = options.Value;
        _detectionService = detectionService;
        _db = db;
    }

    public object GetHoldoutUsers()
    {
        var indexPath = Path.Combine(_paths.ResolvePath(_settings.HoldoutDirectory), "index.json");

        if (!File.Exists(indexPath))
        {
            return new
            {
                users = Array.Empty<object>(),
                warning = $"Holdout index not found: {indexPath}. Run ml/multiview_prepare_fast.py first."
            };
        }

        var json = File.ReadAllText(indexPath);
        return JsonSerializer.Deserialize<object>(json) ?? new { users = Array.Empty<object>() };
    }

    public async Task<DetectionResponseDto> ReplayAsync(DemoReplayRequest request, CancellationToken cancellationToken = default)
    {
        if (request.Count <= 0)
        {
            throw new ArgumentException("Count must be > 0.");
        }

        var holdoutDir = _paths.ResolvePath(_settings.HoldoutDirectory);
        var csvPath = Path.Combine(holdoutDir, $"{SanitizePathPart(request.UserId)}.csv");

        if (!File.Exists(csvPath))
        {
            throw new FileNotFoundException($"Holdout CSV not found for user {request.UserId}: {csvPath}");
        }

        var rows = ReadCsv(csvPath);
        var unit = string.IsNullOrWhiteSpace(request.Unit) ? "days" : request.Unit.Trim().ToLowerInvariant();

        var selectedRows = unit == "events"
            ? await SelectNextEventsAsync(request.UserId, rows, request.Count, cancellationToken)
            : await SelectNextDaysAsync(request.UserId, rows, request.Count, cancellationToken);

        if (selectedRows.Count == 0)
        {
            throw new InvalidOperationException($"No more holdout events to replay for user {request.UserId}.");
        }

        DetectionResponseDto? last = null;

        for (var i = 0; i < selectedRows.Count; i++)
        {
            var dto = MapRowToLogInput(selectedRows[i], request.UserId);
            var shouldDetect = i == selectedRows.Count - 1;

            if (shouldDetect)
            {
                last = await _detectionService.IngestAndDetectAsync(dto, "holdout_replay", allowTraining: true, cancellationToken);
            }
            else
            {
                await _detectionService.InsertWithoutPredictionAsync(dto, "holdout_replay", cancellationToken);
            }
        }

        return last ?? throw new InvalidOperationException("Replay failed.");
    }

    private async Task<List<Dictionary<string, string>>> SelectNextEventsAsync(string userId, List<Dictionary<string, string>> rows, int count, CancellationToken cancellationToken)
    {
        var startOffset = await _db.RawLogs.CountAsync(x => x.UserId == userId, cancellationToken);
        return rows.Skip(startOffset).Take(count).ToList();
    }

    private async Task<List<Dictionary<string, string>>> SelectNextDaysAsync(string userId, List<Dictionary<string, string>> rows, int days, CancellationToken cancellationToken)
    {
        var alreadyDays = await _db.RawLogs
            .Where(x => x.UserId == userId)
            .Select(x => x.TimestampUtc.Date)
            .Distinct()
            .CountAsync(cancellationToken);

        var grouped = rows
            .Select(r => new { Row = r, Day = ExtractDate(r) })
            .Where(x => x.Day != null)
            .GroupBy(x => x.Day!.Value.Date)
            .OrderBy(g => g.Key)
            .ToList();

        return grouped
            .Skip(alreadyDays)
            .Take(days)
            .SelectMany(g => g.Select(x => x.Row))
            .ToList();
    }

    private static DateTime? ExtractDate(Dictionary<string, string> row)
    {
        var timestampRaw = Get(row, "timestamp", "date", "TimestampUtc", "timestamp_utc");
        if (DateTimeOffset.TryParse(timestampRaw, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var timestamp))
        {
            return timestamp.UtcDateTime.Date;
        }
        return null;
    }

    private static LogInputDto MapRowToLogInput(Dictionary<string, string> row, string forcedUserId)
    {
        var eventType = Get(row, "source_file", "event_type", "source", "type");
        if (string.IsNullOrWhiteSpace(eventType))
        {
            eventType = InferEventType(row);
        }

        var timestampRaw = Get(row, "timestamp", "date", "TimestampUtc", "timestamp_utc");
        DateTimeOffset timestamp;

        if (!DateTimeOffset.TryParse(timestampRaw, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out timestamp))
        {
            timestamp = DateTimeOffset.UtcNow;
        }

        return new LogInputDto(
            forcedUserId,
            Get(row, "pc", "pc_id", "PcId"),
            timestamp,
            eventType,
            Get(row, "activity", "event_type"),
            Get(row, "url", "object"),
            Get(row, "filename", "file_name", "object"),
            Get(row, "to", "email_to", "object"),
            Get(row, "cc", "email_cc"),
            Get(row, "bcc", "email_bcc"),
            Get(row, "from", "email_from"),
            TryLong(Get(row, "size")),
            TryInt(Get(row, "attachment_count")),
            Get(row, "content", "source_payload"),
            Get(row, "event_uid", "id", "source_event_id", "original_id")
        );
    }

    private static string InferEventType(Dictionary<string, string> row)
    {
        if (!string.IsNullOrWhiteSpace(Get(row, "url"))) return "http";
        if (!string.IsNullOrWhiteSpace(Get(row, "filename"))) return "file";
        if (!string.IsNullOrWhiteSpace(Get(row, "to", "from"))) return "email";
        if (!string.IsNullOrWhiteSpace(Get(row, "activity"))) return "logon";
        return "unknown";
    }

    private static string? Get(Dictionary<string, string> row, params string[] keys)
    {
        foreach (var key in keys)
        {
            if (row.TryGetValue(key, out var value) && !string.IsNullOrWhiteSpace(value))
            {
                return value;
            }
        }

        return null;
    }

    private static long? TryLong(string? value) => long.TryParse(value, out var parsed) ? parsed : null;

    private static int? TryInt(string? value) => int.TryParse(value, out var parsed) ? parsed : null;

    private static List<Dictionary<string, string>> ReadCsv(string path)
    {
        using var reader = new StreamReader(path);
        var headerLine = reader.ReadLine();
        if (headerLine == null) return new List<Dictionary<string, string>>();

        var headers = ParseCsvLine(headerLine);
        var rows = new List<Dictionary<string, string>>();

        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            if (string.IsNullOrWhiteSpace(line)) continue;
            var values = ParseCsvLine(line);
            var dict = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            for (var i = 0; i < headers.Count; i++) dict[headers[i]] = i < values.Count ? values[i] : string.Empty;
            rows.Add(dict);
        }
        return rows;
    }

    private static List<string> ParseCsvLine(string line)
    {
        var values = new List<string>();
        var current = new System.Text.StringBuilder();
        var inQuotes = false;
        for (var i = 0; i < line.Length; i++)
        {
            var ch = line[i];
            if (ch == '"')
            {
                if (inQuotes && i + 1 < line.Length && line[i + 1] == '"') { current.Append('"'); i++; }
                else inQuotes = !inQuotes;
            }
            else if (ch == ',' && !inQuotes) { values.Add(current.ToString()); current.Clear(); }
            else current.Append(ch);
        }
        values.Add(current.ToString());
        return values;
    }

    private static string SanitizePathPart(string value)
    {
        var invalid = Path.GetInvalidFileNameChars();
        return new string(value.Select(ch => invalid.Contains(ch) ? '_' : ch).ToArray());
    }
}
