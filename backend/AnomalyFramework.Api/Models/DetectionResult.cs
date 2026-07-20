namespace AnomalyFramework.Api.Models;

public class DetectionResult
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public Guid RawLogId { get; set; }

    public RawLog? RawLog { get; set; }

    public string UserId { get; set; } = string.Empty;

    public DateTime TimestampUtc { get; set; }

    public string BaselineRoute { get; set; } = "global";

    public string? Role { get; set; }

    public string? Department { get; set; }

    public double Score { get; set; }

    public double Threshold { get; set; }

    public double? GlobalRatio { get; set; }

    public double? RoleRatio { get; set; }

    public double? PersonalRatio { get; set; }

    public double? FinalAnomalyIndex { get; set; }

    public bool IsAnomaly { get; set; }

    public int LogCountAtPrediction { get; set; }

    public int ActiveDaysAtPrediction { get; set; }

    public bool PersonalizedReadyAtPrediction { get; set; }

    public bool PersonalizedTrainingTriggered { get; set; }

    public bool PersonalizedTrainingCompleted { get; set; }

    public string? ModelVersion { get; set; }

    public string? Warning { get; set; }

    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
}
