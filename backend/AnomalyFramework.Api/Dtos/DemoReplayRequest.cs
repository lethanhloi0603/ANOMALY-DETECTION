namespace AnomalyFramework.Api.Dtos;

public record DemoReplayRequest(
    string UserId,
    int Count,
    string? Unit
);
