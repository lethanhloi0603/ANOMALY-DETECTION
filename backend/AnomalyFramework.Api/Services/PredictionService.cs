using System.Text.Json;
using Microsoft.Extensions.Options;

namespace AnomalyFramework.Api.Services;

public class PredictionService
{
    private readonly PythonRunner _pythonRunner;
    private readonly RuntimePaths _paths;
    private readonly MlSettings _settings;

    public PredictionService(PythonRunner pythonRunner, RuntimePaths paths, IOptions<MlSettings> options)
    {
        _pythonRunner = pythonRunner;
        _paths = paths;
        _settings = options.Value;
    }

    public async Task<MlPrediction> PredictAsync(string userId, string scope, CancellationToken cancellationToken = default)
    {
        var modelRoot = _paths.ResolvePath(_settings.ModelDirectory);
        var modelDir = scope == "personalized"
            ? Path.Combine(modelRoot, "personalized", SanitizePathPart(userId))
            : Path.Combine(modelRoot, "global");
        var globalModelDir = Path.Combine(modelRoot, "global");

        var args = new List<string>
        {
            "--db", _paths.DbPath,
            "--user-id", userId,
            "--scope", scope,
            "--model-dir", modelDir,
            "--global-model-dir", globalModelDir,
            "--role-context", _paths.ResolvePath(_settings.RoleContextPath),
            "--window-size", _settings.WindowSize.ToString()
        };

        var run = await _pythonRunner.RunAsync("predict_multiview.py", args, _settings.PredictionTimeoutSeconds, cancellationToken);

        if (run.ExitCode != 0)
        {
            return new MlPrediction
            {
                Score = 0,
                Threshold = 0,
                IsAnomaly = false,
                ModelVersion = "prediction_failed",
                Warning = $"Prediction worker failed: {run.Stderr} {run.Stdout}"
            };
        }

        return ParsePrediction(run.Stdout);
    }

    private static MlPrediction ParsePrediction(string stdout)
    {
        var jsonLine = stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .LastOrDefault(line => line.StartsWith("{") && line.EndsWith("}"));

        if (jsonLine == null)
        {
            return new MlPrediction
            {
                Score = 0,
                Threshold = 0,
                IsAnomaly = false,
                ModelVersion = "prediction_parse_failed",
                Warning = $"Could not parse prediction stdout: {stdout}"
            };
        }

        return JsonSerializer.Deserialize<MlPrediction>(jsonLine, new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        }) ?? new MlPrediction
        {
            Score = 0,
            Threshold = 0,
            IsAnomaly = false,
            ModelVersion = "prediction_empty",
            Warning = "Empty prediction result."
        };
    }

    private static string SanitizePathPart(string value)
    {
        var invalid = Path.GetInvalidFileNameChars();
        return new string(value.Select(ch => invalid.Contains(ch) ? '_' : ch).ToArray());
    }
}
