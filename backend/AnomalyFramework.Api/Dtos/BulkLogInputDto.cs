namespace AnomalyFramework.Api.Dtos;

public record BulkLogInputDto(
    string UserId,
    string? PcId,
    int Count,
    DateTimeOffset? StartTime,
    bool ScoreOnlyLast,
    List<LogInputDto>? Logs
);
