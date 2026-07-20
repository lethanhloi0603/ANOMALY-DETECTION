using AnomalyFramework.Api.Models;
using Microsoft.EntityFrameworkCore;

namespace AnomalyFramework.Api.Data;

public class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
    {
    }

    public DbSet<RawLog> RawLogs => Set<RawLog>();

    public DbSet<UserModelState> UserModelStates => Set<UserModelState>();

    public DbSet<DetectionResult> DetectionResults => Set<DetectionResult>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<RawLog>(entity =>
        {
            entity.HasKey(x => x.Id);
            entity.HasIndex(x => x.UserId);
            entity.HasIndex(x => x.TimestampUtc);
            entity.HasIndex(x => new { x.UserId, x.TimestampUtc });
            entity.Property(x => x.UserId).HasMaxLength(128);
            entity.Property(x => x.PcId).HasMaxLength(128);
            entity.Property(x => x.EventType).HasMaxLength(64);
            entity.Property(x => x.Activity).HasMaxLength(128);
            entity.Property(x => x.InputMode).HasMaxLength(64);
        });

        modelBuilder.Entity<UserModelState>(entity =>
        {
            entity.HasKey(x => x.Id);
            entity.HasIndex(x => x.UserId).IsUnique();
            entity.Property(x => x.UserId).HasMaxLength(128);
            entity.Property(x => x.Role).HasMaxLength(128);
            entity.Property(x => x.Department).HasMaxLength(256);
        });

        modelBuilder.Entity<DetectionResult>(entity =>
        {
            entity.HasKey(x => x.Id);
            entity.HasIndex(x => x.UserId);
            entity.HasIndex(x => x.TimestampUtc);
            entity.Property(x => x.BaselineRoute).HasMaxLength(32);
            entity.Property(x => x.Role).HasMaxLength(128);
            entity.Property(x => x.Department).HasMaxLength(256);
        });
    }
}
