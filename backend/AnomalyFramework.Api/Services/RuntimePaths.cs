using Microsoft.Data.Sqlite;

namespace AnomalyFramework.Api.Services;

public class RuntimePaths
{
    public RuntimePaths(string contentRootPath, string rawConnectionString)
    {
        ContentRootPath = contentRootPath;
        DbPath = ResolveSqliteDataSource(contentRootPath, rawConnectionString);
    }

    public string ContentRootPath { get; }
    public string DbPath { get; }

    public string ResolvePath(string relativeOrAbsolutePath)
    {
        if (Path.IsPathRooted(relativeOrAbsolutePath)) return Path.GetFullPath(relativeOrAbsolutePath);
        return Path.GetFullPath(Path.Combine(ContentRootPath, relativeOrAbsolutePath));
    }

    private static string ResolveSqliteDataSource(string contentRootPath, string connectionString)
    {
        var builder = new SqliteConnectionStringBuilder(connectionString);
        var dataSource = string.IsNullOrWhiteSpace(builder.DataSource) ? "../../data/anomaly.db" : builder.DataSource;
        var fullPath = Path.IsPathRooted(dataSource) ? Path.GetFullPath(dataSource) : Path.GetFullPath(Path.Combine(contentRootPath, dataSource));
        var dir = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrWhiteSpace(dir)) Directory.CreateDirectory(dir);
        return fullPath;
    }
}
