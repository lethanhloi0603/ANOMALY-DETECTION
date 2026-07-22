using AnomalyFramework.Api.Data;
using AnomalyFramework.Api.Dtos;
using AnomalyFramework.Api.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace AnomalyFramework.Api.Services;

public class DetectionService
{
    private readonly AppDbContext _db;
    private readonly PredictionService _predictionService;
    private readonly TrainingService _trainingService;
    private readonly MlSettings _settings;

    public DetectionService(
        AppDbContext db,
        PredictionService predictionService,
        TrainingService trainingService,
        IOptions<MlSettings> options)
    {
        _db = db;
        _predictionService = predictionService;
        _trainingService = trainingService;
        _settings = options.Value;
    }

    public async Task<DetectionResponseDto> IngestAndDetectAsync(
        LogInputDto input,
        string inputMode = "manual",
        bool allowTraining = true,
        CancellationToken cancellationToken = default)
    {
        ValidateInput(input);

        var raw = new RawLog
        {
            SourceEventId = input.SourceEventId,
            UserId = input.UserId.Trim(),
            PcId = input.PcId,
            TimestampUtc = (input.Timestamp ?? DateTimeOffset.UtcNow).UtcDateTime,
            EventType = input.EventType.Trim().ToLowerInvariant(),
            Activity = input.Activity,
            Url = input.Url,
            FileName = input.FileName,
            EmailTo = input.EmailTo,
            EmailCc = input.EmailCc,
            EmailBcc = input.EmailBcc,
            EmailFrom = input.EmailFrom,
            Size = input.Size,
            AttachmentCount = input.AttachmentCount,
            Content = input.Content,
            InputMode = inputMode
        };

        _db.RawLogs.Add(raw);
        await _db.SaveChangesAsync(cancellationToken);

        var logCount = await _db.RawLogs.CountAsync(x => x.UserId == raw.UserId, cancellationToken);
        var activeDays = await CountActiveDaysAsync(raw.UserId, cancellationToken);
        var state = await UpsertUserStateAsync(raw.UserId, logCount, activeDays, cancellationToken);

        var route = state.IsPersonalizedReady ? "personalized" : "global";
        var prediction = await _predictionService.PredictAsync(raw.UserId, route, cancellationToken);

        var trainingTriggered = false;
        var trainingCompleted = false;

        // Active days are only a cheap pre-check. The Python worker activates or
        // updates a personal threshold only when enough delayed safe days exist.
        // The neural model remains global; only the robust personal calibration is updated.
        if (allowTraining && activeDays >= _settings.PersonalizedActiveDaysThreshold)
        {
            trainingTriggered = true;
            trainingCompleted = await _trainingService.TrainPersonalizedIfEligibleAsync(raw.UserId, activeDays, cancellationToken);
            state = await UpsertUserStateAsync(raw.UserId, logCount, activeDays, cancellationToken);
        }

        if (!string.IsNullOrWhiteSpace(prediction.Role) || !string.IsNullOrWhiteSpace(prediction.Department))
        {
            state.Role = prediction.Role;
            state.Department = prediction.Department;
            state.UpdatedAtUtc = DateTime.UtcNow;
            await _db.SaveChangesAsync(cancellationToken);
        }

        var detection = new DetectionResult
        {
            RawLogId = raw.Id,
            UserId = raw.UserId,
            TimestampUtc = raw.TimestampUtc,
            BaselineRoute = route,
            Role = prediction.Role,
            Department = prediction.Department,
            Score = prediction.Score,
            Threshold = prediction.Threshold,
            GlobalRatio = prediction.GlobalRatio,
            RoleRatio = prediction.RoleRatio,
            PersonalRatio = prediction.PersonalRatio,
            FinalAnomalyIndex = prediction.FinalAnomalyIndex,
            IsAnomaly = prediction.IsAnomaly,
            LogCountAtPrediction = logCount,
            ActiveDaysAtPrediction = activeDays,
            PersonalizedReadyAtPrediction = state.IsPersonalizedReady,
            PersonalizedTrainingTriggered = trainingTriggered,
            PersonalizedTrainingCompleted = trainingCompleted,
            ModelVersion = prediction.ModelVersion,
            Warning = prediction.Warning
        };

        _db.DetectionResults.Add(detection);
        await _db.SaveChangesAsync(cancellationToken);

        return new DetectionResponseDto(
            raw.Id,
            detection.Id,
            raw.UserId,
            logCount,
            activeDays,
            route,
            prediction.Role,
            prediction.Department,
            state.IsPersonalizedReady,
            trainingTriggered,
            trainingCompleted,
            prediction.Score,
            prediction.Threshold,
            prediction.GlobalRatio,
            prediction.RoleRatio,
            prediction.PersonalRatio,
            prediction.FinalAnomalyIndex,
            prediction.IsAnomaly,
            prediction.ModelVersion,
            prediction.Warning
        );
    }

    public async Task<DetectionResponseDto> IngestBulkAsync(BulkLogInputDto input, CancellationToken cancellationToken = default)
    {
        if (input.Count <= 0 && (input.Logs == null || input.Logs.Count == 0))
        {
            throw new ArgumentException("Bulk count or logs must be provided.");
        }

        var logs = input.Logs is { Count: > 0 }
            ? input.Logs
            : GenerateSyntheticLogs(input);

        DetectionResponseDto? last = null;

        for (var i = 0; i < logs.Count; i++)
        {
            var shouldScore = !input.ScoreOnlyLast || i == logs.Count - 1;
            if (shouldScore)
            {
                last = await IngestAndDetectAsync(logs[i], "bulk", allowTraining: true, cancellationToken);
            }
            else
            {
                await InsertWithoutPredictionAsync(logs[i], "bulk", cancellationToken);
            }
        }

        if (last == null)
        {
            var final = logs[^1];
            last = await IngestAndDetectAsync(final, "bulk", allowTraining: true, cancellationToken);
        }

        return last;
    }

    public async Task InsertWithoutPredictionAsync(LogInputDto input, string inputMode, CancellationToken cancellationToken = default)
    {
        ValidateInput(input);

        var raw = new RawLog
        {
            SourceEventId = input.SourceEventId,
            UserId = input.UserId.Trim(),
            PcId = input.PcId,
            TimestampUtc = (input.Timestamp ?? DateTimeOffset.UtcNow).UtcDateTime,
            EventType = input.EventType.Trim().ToLowerInvariant(),
            Activity = input.Activity,
            Url = input.Url,
            FileName = input.FileName,
            EmailTo = input.EmailTo,
            EmailCc = input.EmailCc,
            EmailBcc = input.EmailBcc,
            EmailFrom = input.EmailFrom,
            Size = input.Size,
            AttachmentCount = input.AttachmentCount,
            Content = input.Content,
            InputMode = inputMode
        };

        _db.RawLogs.Add(raw);
        await _db.SaveChangesAsync(cancellationToken);

        var logCount = await _db.RawLogs.CountAsync(x => x.UserId == raw.UserId, cancellationToken);
        var activeDays = await CountActiveDaysAsync(raw.UserId, cancellationToken);
        await UpsertUserStateAsync(raw.UserId, logCount, activeDays, cancellationToken);
    }

    private async Task<int> CountActiveDaysAsync(string userId, CancellationToken cancellationToken)
    {
        return await _db.RawLogs
            .Where(x => x.UserId == userId)
            .Select(x => x.TimestampUtc.Date)
            .Distinct()
            .CountAsync(cancellationToken);
    }

    private async Task<UserModelState> UpsertUserStateAsync(string userId, int logCount, int activeDays, CancellationToken cancellationToken)
    {
        var state = await _db.UserModelStates.FirstOrDefaultAsync(x => x.UserId == userId, cancellationToken);
        if (state == null)
        {
            state = new UserModelState { UserId = userId };
            _db.UserModelStates.Add(state);
        }

        state.LogCount = logCount;
        state.ActiveDaysCount = activeDays;
        state.UpdatedAtUtc = DateTime.UtcNow;

        await _db.SaveChangesAsync(cancellationToken);
        return state;
    }

    private static List<LogInputDto> GenerateSyntheticLogs(BulkLogInputDto input)
    {
        var result = new List<LogInputDto>();
        var start = input.StartTime ?? DateTimeOffset.UtcNow.AddDays(-Math.Max(input.Count, 1));
        var eventTypes = new[] { "logon", "http", "email", "file", "device" };

        for (var i = 0; i < input.Count; i++)
        {
            var type = eventTypes[i % eventTypes.Length];
            // Mỗi log cách nhau 1 ngày để demo ngưỡng 30 active days nhanh hơn.
            var timestamp = start.AddDays(i);
            var activity = type switch
            {
                "logon" => "Logon",
                "device" => i % 2 == 0 ? "connect" : "disconnect",
                "file" => "copy",
                "email" => "send",
                "http" => "visit",
                _ => "activity"
            };

            result.Add(new LogInputDto(
                input.UserId,
                input.PcId ?? $"PC-{input.UserId}",
                timestamp,
                type,
                activity,
                type == "http" ? "https://example.com/research/security" : null,
                type == "file" ? $"document_{i}.docx" : null,
                type == "email" ? "colleague@dtaa.com" : null,
                null,
                null,
                type == "email" ? $"{input.UserId}@dtaa.com" : null,
                type == "email" ? 2000 + i : null,
                type == "email" ? i % 3 : null,
                $"synthetic demo content {type} #{i}",
                $"synthetic-{input.UserId}-{i}"
            ));
        }

        return result;
    }

    private static void ValidateInput(LogInputDto input)
    {
        if (string.IsNullOrWhiteSpace(input.UserId))
        {
            throw new ArgumentException("UserId is required.");
        }

        if (string.IsNullOrWhiteSpace(input.EventType))
        {
            throw new ArgumentException("EventType is required.");
        }
    }
}
