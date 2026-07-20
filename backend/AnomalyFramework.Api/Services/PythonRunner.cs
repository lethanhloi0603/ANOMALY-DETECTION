using System.Diagnostics;
using System.Text;
using Microsoft.Extensions.Options;

namespace AnomalyFramework.Api.Services;

public record PythonRunResult(int ExitCode, string Stdout, string Stderr);

public class PythonRunner
{
    private readonly RuntimePaths _paths;
    private readonly MlSettings _settings;
    private readonly ILogger<PythonRunner> _logger;

    public PythonRunner(RuntimePaths paths, IOptions<MlSettings> options, ILogger<PythonRunner> logger)
    {
        _paths = paths;
        _settings = options.Value;
        _logger = logger;
    }

    public async Task<PythonRunResult> RunAsync(string scriptName, IEnumerable<string> args, int timeoutSeconds, CancellationToken cancellationToken = default)
    {
        var workingDirectory = _paths.ResolvePath(_settings.WorkingDirectory);
        var scriptPath = Path.Combine(workingDirectory, scriptName);
        if (!File.Exists(scriptPath)) throw new FileNotFoundException($"Python script not found: {scriptPath}");

        var psi = new ProcessStartInfo
        {
            FileName = _settings.PythonExecutable,
            WorkingDirectory = workingDirectory,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };

        psi.ArgumentList.Add(scriptPath);
        foreach (var arg in args) psi.ArgumentList.Add(arg);

        using var process = new Process { StartInfo = psi };
        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        process.OutputDataReceived += (_, e) => { if (e.Data != null) stdout.AppendLine(e.Data); };
        process.ErrorDataReceived += (_, e) => { if (e.Data != null) stderr.AppendLine(e.Data); };

        _logger.LogInformation("Running Python: {FileName} {Args}", psi.FileName, string.Join(" ", psi.ArgumentList));
        process.Start();
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();

        using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeoutCts.CancelAfter(TimeSpan.FromSeconds(timeoutSeconds));
        try
        {
            await process.WaitForExitAsync(timeoutCts.Token);
        }
        catch (OperationCanceledException)
        {
            try { if (!process.HasExited) process.Kill(entireProcessTree: true); } catch { }
            throw new TimeoutException($"Python script timeout after {timeoutSeconds}s: {scriptName}");
        }

        return new PythonRunResult(process.ExitCode, stdout.ToString(), stderr.ToString());
    }
}
