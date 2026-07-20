namespace AnomalyFramework.Api.Models;

public class RawLog
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string? SourceEventId { get; set; }
    public string UserId { get; set; } = string.Empty;
    public string? PcId { get; set; }
    public DateTime TimestampUtc { get; set; }
    public string EventType { get; set; } = string.Empty;
    public string? Activity { get; set; }
    public string? Url { get; set; }
    public string? FileName { get; set; }
    public string? EmailTo { get; set; }
    public string? EmailCc { get; set; }
    public string? EmailBcc { get; set; }
    public string? EmailFrom { get; set; }
    public long? Size { get; set; }
    public int? AttachmentCount { get; set; }
    public string? Content { get; set; }
    public string InputMode { get; set; } = "manual";
    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
}
