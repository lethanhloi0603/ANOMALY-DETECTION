namespace AnomalyFramework.Api.Services;

public class MlSettings
{
    public string PythonExecutable { get; set; } = "python";

    public string WorkingDirectory { get; set; } = "../../ml";

    public string ModelDirectory { get; set; } = "../../models";

    public string GlobalTrainingMultiviewPath { get; set; } = "../../data/processed/03_user_day_multiview.csv";

    public string RoleContextPath { get; set; } = "../../data/processed/role_context.csv";

    public string HoldoutDirectory { get; set; } = "../../data/processed/holdout";

    public string FeatureRulesPath { get; set; } = "../../ml/feature_rules.json";

    public int PersonalizedActiveDaysThreshold { get; set; } = 30;

    public int WindowSize { get; set; } = 30;

    public int TrainEpochs { get; set; } = 3;

    public int TrainBatchSize { get; set; } = 32;

    public double LearningRate { get; set; } = 0.0005;

    public int HiddenDimension { get; set; } = 128;

    public int KernelSize { get; set; } = 3;

    public double Dropout { get; set; } = 0.15;

    public int TransformerLayers { get; set; } = 2;

    public int AttentionHeads { get; set; } = 4;

    public int MaxEventsPerDay { get; set; } = 256;

    public string TrainingLabelPolicy { get; set; } = "ignore";

    public string CalibrationLabelPolicy { get; set; } = "ignore";

    public int TrainingNumWorkers { get; set; } = 0;

    public int TrainingProgressEvery { get; set; } = 250;

    public int RandomSeed { get; set; } = 42;

    public double AnomalyQuantile { get; set; } = 0.995;

    public int MinRoleUsers { get; set; } = 30;

    public int MinRoleUserDays { get; set; } = 1000;

    public int PersonalizedSafeHistoryDays { get; set; } = 90;

    public int PersonalizedUpdateDelayDays { get; set; } = 7;

    public int PersonalizedUpdateEverySafeDays { get; set; } = 7;

    public int PersonalizedUpdateEveryCalendarDays { get; set; } = 7;

    public int PredictionTimeoutSeconds { get; set; } = 120;

    public int TrainingTimeoutSeconds { get; set; } = 21600;
}
