using System.Text.Json.Serialization;

namespace AnomalyFramework.Api.Services;

public class MlPrediction
{
    [JsonPropertyName("score")]
    public double Score { get; set; }

    [JsonPropertyName("threshold")]
    public double Threshold { get; set; }

    [JsonPropertyName("is_anomaly")]
    public bool IsAnomaly { get; set; }

    [JsonPropertyName("model_version")]
    public string? ModelVersion { get; set; }

    [JsonPropertyName("role")]
    public string? Role { get; set; }

    [JsonPropertyName("department")]
    public string? Department { get; set; }

    [JsonPropertyName("global_ratio")]
    public double? GlobalRatio { get; set; }

    [JsonPropertyName("role_ratio")]
    public double? RoleRatio { get; set; }

    [JsonPropertyName("personal_ratio")]
    public double? PersonalRatio { get; set; }

    [JsonPropertyName("final_anomaly_index")]
    public double? FinalAnomalyIndex { get; set; }

    [JsonPropertyName("warning")]
    public string? Warning { get; set; }
}

public class MlTrainingResult
{
    [JsonPropertyName("ok")]
    public bool Ok { get; set; }

    [JsonPropertyName("eligible")]
    public bool Eligible { get; set; }

    [JsonPropertyName("updated")]
    public bool Updated { get; set; }

    [JsonPropertyName("scope")]
    public string? Scope { get; set; }

    [JsonPropertyName("user_id")]
    public string? UserId { get; set; }

    [JsonPropertyName("model_dir")]
    public string? ModelDir { get; set; }

    [JsonPropertyName("windows")]
    public int Windows { get; set; }

    [JsonPropertyName("events")]
    public int Events { get; set; }

    [JsonPropertyName("active_days")]
    public int ActiveDays { get; set; }

    [JsonPropertyName("safe_days")]
    public int SafeDays { get; set; }

    [JsonPropertyName("threshold")]
    public double Threshold { get; set; }

    [JsonPropertyName("message")]
    public string? Message { get; set; }
}
