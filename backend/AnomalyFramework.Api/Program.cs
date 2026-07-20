using AnomalyFramework.Api.Data;
using AnomalyFramework.Api.Dtos;
using AnomalyFramework.Api.Services;
using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);

var rawConnectionString = builder.Configuration.GetConnectionString("Default") ?? "Data Source=../../data/anomaly.db";
var runtimePaths = new RuntimePaths(builder.Environment.ContentRootPath, rawConnectionString);

builder.Services.Configure<MlSettings>(builder.Configuration.GetSection("Ml"));
builder.Services.AddSingleton(runtimePaths);
builder.Services.AddScoped<PythonRunner>();
builder.Services.AddScoped<PredictionService>();
builder.Services.AddScoped<TrainingService>();
builder.Services.AddScoped<DetectionService>();
builder.Services.AddScoped<DemoReplayService>();

builder.Services.AddDbContext<AppDbContext>(options =>
{
    options.UseSqlite($"Data Source={runtimePaths.DbPath}");
});

builder.Services.AddCors(options =>
{
    options.AddPolicy("frontend", policy =>
    {
        policy
            .WithOrigins("http://localhost:5173", "http://127.0.0.1:5173")
            .AllowAnyHeader()
            .AllowAnyMethod();
    });
});

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();
    db.Database.EnsureCreated();
}

app.UseCors("frontend");

if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.MapGet("/", () => Results.Ok(new
{
    name = "CERT Insider Threat Multi-view Anomaly Framework API",
    database = runtimePaths.DbPath,
    swagger = "/swagger"
}));

app.MapPost("/api/logs", async (LogInputDto input, DetectionService detectionService, CancellationToken cancellationToken) =>
{
    var result = await detectionService.IngestAndDetectAsync(input, "manual", allowTraining: true, cancellationToken);
    return Results.Ok(result);
});

app.MapPost("/api/logs/bulk", async (BulkLogInputDto input, DetectionService detectionService, CancellationToken cancellationToken) =>
{
    var result = await detectionService.IngestBulkAsync(input, cancellationToken);
    return Results.Ok(result);
});

app.MapGet("/api/users/{userId}/status", async (string userId, AppDbContext db, CancellationToken cancellationToken) =>
{
    var logCount = await db.RawLogs.CountAsync(x => x.UserId == userId, cancellationToken);
    var activeDays = await db.RawLogs.Where(x => x.UserId == userId).Select(x => x.TimestampUtc.Date).Distinct().CountAsync(cancellationToken);
    var state = await db.UserModelStates.FirstOrDefaultAsync(x => x.UserId == userId, cancellationToken);

    return Results.Ok(new
    {
        userId,
        logCount,
        activeDaysCount = activeDays,
        role = state?.Role,
        department = state?.Department,
        personalizedReady = state?.IsPersonalizedReady ?? false,
        isTraining = state?.IsTraining ?? false,
        personalizedModelPath = state?.PersonalizedModelPath,
        lastPersonalizedTrainingAtUtc = state?.LastPersonalizedTrainingAtUtc,
        lastTrainingMessage = state?.LastTrainingMessage
    });
});

app.MapGet("/api/detections", async (string? userId, AppDbContext db, CancellationToken cancellationToken) =>
{
    var query = db.DetectionResults.OrderByDescending(x => x.CreatedAtUtc).AsQueryable();
    if (!string.IsNullOrWhiteSpace(userId)) query = query.Where(x => x.UserId == userId);

    var results = await query.Take(50).Select(x => new
    {
        x.Id,
        x.RawLogId,
        x.UserId,
        x.TimestampUtc,
        x.BaselineRoute,
        x.Role,
        x.Department,
        x.Score,
        x.Threshold,
        x.GlobalRatio,
        x.RoleRatio,
        x.PersonalRatio,
        x.FinalAnomalyIndex,
        x.IsAnomaly,
        x.LogCountAtPrediction,
        x.ActiveDaysAtPrediction,
        x.PersonalizedReadyAtPrediction,
        x.PersonalizedTrainingTriggered,
        x.PersonalizedTrainingCompleted,
        x.ModelVersion,
        x.Warning,
        x.CreatedAtUtc
    }).ToListAsync(cancellationToken);

    return Results.Ok(results);
});

app.MapGet("/api/dashboard/summary", async (AppDbContext db, CancellationToken cancellationToken) =>
{
    var totalLogs = await db.RawLogs.CountAsync(cancellationToken);
    var totalUsers = await db.RawLogs.Select(x => x.UserId).Distinct().CountAsync(cancellationToken);
    var personalizedReadyUsers = await db.UserModelStates.CountAsync(x => x.IsPersonalizedReady, cancellationToken);
    var totalDetections = await db.DetectionResults.CountAsync(cancellationToken);
    var anomalies = await db.DetectionResults.CountAsync(x => x.IsAnomaly, cancellationToken);
    var globalRoutes = await db.DetectionResults.CountAsync(x => x.BaselineRoute == "global", cancellationToken);
    var personalizedRoutes = await db.DetectionResults.CountAsync(x => x.BaselineRoute == "personalized", cancellationToken);
    var activeDays = await db.RawLogs.Select(x => new { x.UserId, Day = x.TimestampUtc.Date }).Distinct().CountAsync(cancellationToken);

    return Results.Ok(new { totalLogs, totalUsers, activeDays, personalizedReadyUsers, totalDetections, anomalies, globalRoutes, personalizedRoutes });
});

app.MapPost("/api/training/global", async (TrainingService trainingService, CancellationToken cancellationToken) =>
{
    var result = await trainingService.TrainGlobalAsync(cancellationToken);
    return Results.Ok(result);
});

app.MapPost("/api/training/personalized/{userId}", async (string userId, TrainingService trainingService, CancellationToken cancellationToken) =>
{
    var result = await trainingService.TrainPersonalizedAsync(userId, cancellationToken);
    return Results.Ok(result);
});

app.MapGet("/api/demo/holdout-users", (DemoReplayService demoReplayService) => Results.Ok(demoReplayService.GetHoldoutUsers()));

app.MapPost("/api/demo/replay", async (DemoReplayRequest request, DemoReplayService demoReplayService, CancellationToken cancellationToken) =>
{
    var result = await demoReplayService.ReplayAsync(request, cancellationToken);
    return Results.Ok(result);
});

app.Run();
