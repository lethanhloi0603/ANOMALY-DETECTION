using System.Text.Json;
using AnomalyFramework.Api.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;

namespace AnomalyFramework.Api.Services;

public class TrainingService
{
    private readonly AppDbContext _db;
    private readonly PythonRunner _pythonRunner;
    private readonly RuntimePaths _paths;
    private readonly MlSettings _settings;
    private readonly ILogger<TrainingService> _logger;

    public TrainingService(
        AppDbContext db,
        PythonRunner pythonRunner,
        RuntimePaths paths,
        IOptions<MlSettings> options,
        ILogger<TrainingService> logger)
    {
        _db = db;
        _pythonRunner = pythonRunner;
        _paths = paths;
        _settings = options.Value;
        _logger = logger;
    }

    public async Task<MlTrainingResult> TrainGlobalAsync(CancellationToken cancellationToken = default)
    {
        var modelDir = Path.Combine(_paths.ResolvePath(_settings.ModelDirectory), "global");
        Directory.CreateDirectory(modelDir);

        var multiviewPath = _paths.ResolvePath(_settings.GlobalTrainingMultiviewPath);
        var roleContextPath = _paths.ResolvePath(_settings.RoleContextPath);

        var args = new List<string>
        {
            "--scope", "global",
            "--multiview-csv", multiviewPath,
            "--role-context", roleContextPath,
            "--model-dir", modelDir,
            "--window-size", _settings.WindowSize.ToString(),
            "--epochs", _settings.TrainEpochs.ToString(),
            "--anomaly-quantile", _settings.AnomalyQuantile.ToString(System.Globalization.CultureInfo.InvariantCulture),
            "--min-role-samples", _settings.MinRoleSamples.ToString()
        };

        var run = await _pythonRunner.RunAsync("train_multiview.py", args, _settings.TrainingTimeoutSeconds, cancellationToken);

        if (run.ExitCode != 0)
        {
            throw new InvalidOperationException($"Global training failed. STDERR: {run.Stderr} STDOUT: {run.Stdout}");
        }

        return ParseTrainingResult(run.Stdout);
    }

    public async Task<MlTrainingResult> TrainPersonalizedAsync(string userId, CancellationToken cancellationToken = default)
    {
        var state = await _db.UserModelStates.FirstOrDefaultAsync(x => x.UserId == userId, cancellationToken);
        if (state == null)
        {
            state = new Models.UserModelState { UserId = userId };
            _db.UserModelStates.Add(state);
        }

        state.IsTraining = true;
        state.UpdatedAtUtc = DateTime.UtcNow;
        await _db.SaveChangesAsync(cancellationToken);

        try
        {
            var modelRoot = _paths.ResolvePath(_settings.ModelDirectory);
            var modelDir = Path.Combine(modelRoot, "personalized", SanitizePathPart(userId));
            var globalModelDir = Path.Combine(modelRoot, "global");
            Directory.CreateDirectory(modelDir);

            var args = new List<string>
            {
                "--scope", "personalized",
                "--db", _paths.DbPath,
                "--user-id", userId,
                "--model-dir", modelDir,
                "--global-model-dir", globalModelDir,
                "--role-context", _paths.ResolvePath(_settings.RoleContextPath),
                "--window-size", _settings.WindowSize.ToString(),
                "--epochs", _settings.TrainEpochs.ToString(),
                "--min-active-days", _settings.PersonalizedActiveDaysThreshold.ToString(),
                "--anomaly-quantile", _settings.AnomalyQuantile.ToString(System.Globalization.CultureInfo.InvariantCulture)
            };

            var run = await _pythonRunner.RunAsync("train_multiview.py", args, _settings.TrainingTimeoutSeconds, cancellationToken);

            if (run.ExitCode != 0)
            {
                state.IsTraining = false;
                state.LastTrainingMessage = $"FAILED: {run.Stderr} {run.Stdout}";
                state.UpdatedAtUtc = DateTime.UtcNow;
                await _db.SaveChangesAsync(cancellationToken);

                throw new InvalidOperationException($"Personalized training failed for {userId}. STDERR: {run.Stderr} STDOUT: {run.Stdout}");
            }

            var result = ParseTrainingResult(run.Stdout);
            var logCount = await _db.RawLogs.CountAsync(x => x.UserId == userId, cancellationToken);
            var activeDays = await _db.RawLogs
                .Where(x => x.UserId == userId)
                .Select(x => x.TimestampUtc.Date)
                .Distinct()
                .CountAsync(cancellationToken);

            state.LogCount = logCount;
            state.ActiveDaysCount = activeDays;
            state.IsTraining = false;
            state.IsPersonalizedReady = true;
            state.PersonalizedModelPath = modelDir;
            state.LastPersonalizedTrainingAtUtc = DateTime.UtcNow;
            state.LastTrainingMessage = result.Message ?? "Personalized model trained.";
            state.UpdatedAtUtc = DateTime.UtcNow;

            await _db.SaveChangesAsync(cancellationToken);

            return result;
        }
        catch
        {
            state.IsTraining = false;
            state.UpdatedAtUtc = DateTime.UtcNow;
            await _db.SaveChangesAsync(cancellationToken);
            throw;
        }
    }

    public async Task<bool> TrainPersonalizedIfEligibleAsync(string userId, int activeDaysCount, CancellationToken cancellationToken = default)
    {
        var state = await _db.UserModelStates.FirstOrDefaultAsync(x => x.UserId == userId, cancellationToken);

        if (activeDaysCount < _settings.PersonalizedActiveDaysThreshold)
        {
            return false;
        }

        if (state is { IsPersonalizedReady: true })
        {
            return false;
        }

        if (state is { IsTraining: true })
        {
            return false;
        }

        _logger.LogInformation("User {UserId} reached {ActiveDays} active days. Training personalized baseline.", userId, activeDaysCount);

        await TrainPersonalizedAsync(userId, cancellationToken);
        return true;
    }

    private static MlTrainingResult ParseTrainingResult(string stdout)
    {
        var jsonLine = stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .LastOrDefault(line => line.StartsWith("{") && line.EndsWith("}"));

        if (jsonLine == null)
        {
            throw new InvalidOperationException($"Could not parse ML training result from stdout: {stdout}");
        }

        return JsonSerializer.Deserialize<MlTrainingResult>(jsonLine, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        }) ?? throw new InvalidOperationException($"Empty ML training result: {stdout}");
    }

    private static string SanitizePathPart(string value)
    {
        var invalid = Path.GetInvalidFileNameChars();
        return new string(value.Select(ch => invalid.Contains(ch) ? '_' : ch).ToArray());
    }
}
