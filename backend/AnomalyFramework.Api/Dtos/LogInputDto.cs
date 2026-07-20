namespace AnomalyFramework.Api.Dtos;

public record LogInputDto(
    string UserId,
    string? PcId,
    DateTimeOffset? Timestamp,
    string EventType,
    string? Activity,
    string? Url,
    string? FileName,
    string? EmailTo,
    string? EmailCc,
    string? EmailBcc,
    string? EmailFrom,
    long? Size,
    int? AttachmentCount,
    string? Content,
    string? SourceEventId
);
