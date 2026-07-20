namespace AnomalyFramework.Api.Services;

public class MlSettings
{
    public string PythonExecutable { get; set; } = "python";

    public string WorkingDirectory { get; set; } = "../../ml";

    public string ModelDirectory { get; set; } = "../../models";

    public string GlobalTrainingMultiviewPath { get; set; } = "../../data/processed/03_user_day_multiview.csv";

    public string RoleContextPath { get; set; } = "../../data/processed/role_context.csv";

    public string HoldoutDirectory { get; set; } = "../../data/processed/holdout";

    public int PersonalizedActiveDaysThreshold { get; set; } = 30;

    public int WindowSize { get; set; } = 30;

    public int TrainEpochs { get; set; } = 3;

    public double AnomalyQuantile { get; set; } = 0.995;

    public int MinRoleSamples { get; set; } = 25;

    public int PredictionTimeoutSeconds { get; set; } = 120;

    public int TrainingTimeoutSeconds { get; set; } = 900;
}
