-- Apply this file to a database that is physically separate from the core backend.
-- Never grant this database credential to the API, ingestion worker, or scorer.

CREATE TABLE IF NOT EXISTS evaluation_labels (
    user_id VARCHAR(128) NOT NULL,
    day DATE NOT NULL,
    incident_id VARCHAR(255),
    scenario VARCHAR(255),
    answer_metadata JSON NOT NULL,
    source_checksum VARCHAR(128) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, day)
);
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id VARCHAR(36) PRIMARY KEY,
    model_version VARCHAR(128) NOT NULL,
    config_version VARCHAR(128) NOT NULL,
    decision_manifest_checksum VARCHAR(128) NOT NULL,
    label_manifest_checksum VARCHAR(128) NOT NULL,
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_metrics (
    id VARCHAR(36) PRIMARY KEY,
    run_id VARCHAR(36) NOT NULL REFERENCES evaluation_runs(id),
    metric_name VARCHAR(128) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    denominator INTEGER,
    confidence_low DOUBLE PRECISION,
    confidence_high DOUBLE PRECISION,
    dimensions JSON NOT NULL,
    UNIQUE (run_id, metric_name, dimensions)
);
