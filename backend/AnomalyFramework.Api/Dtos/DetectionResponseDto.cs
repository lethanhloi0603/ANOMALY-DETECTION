namespace AnomalyFramework.Api.Dtos;

public record DetectionResponseDto(
    Guid RawLogId,
    Guid DetectionId,
    string UserId,
    int LogCount,
    int ActiveDaysCount,
    string BaselineRoute,
    string? Role,
    string? Department,
    bool PersonalizedReady,
    bool PersonalizedTrainingTriggered,
    bool PersonalizedTrainingCompleted,
    double Score,
    double Threshold,
    double? GlobalRatio,
    double? RoleRatio,
    double? PersonalRatio,
    double? FinalAnomalyIndex,
    bool IsAnomaly,
    string? ModelVersion,
    string? Warning
);
