namespace AnomalyFramework.Api.Models;

public class UserModelState
{
    public Guid Id { get; set; } = Guid.NewGuid();

    public string UserId { get; set; } = string.Empty;

    public int LogCount { get; set; }

    public int ActiveDaysCount { get; set; }

    public string? Role { get; set; }

    public string? Department { get; set; }

    public bool IsPersonalizedReady { get; set; }

    public bool IsTraining { get; set; }

    public string? PersonalizedModelPath { get; set; }

    public DateTime? LastPersonalizedTrainingAtUtc { get; set; }

    public string? LastTrainingMessage { get; set; }

    public DateTime UpdatedAtUtc { get; set; } = DateTime.UtcNow;
}
