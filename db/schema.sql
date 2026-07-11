CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    hashed_pw TEXT,
    role TEXT NOT NULL CHECK(role IN ('admin','editor','viewer')),
    provider TEXT NOT NULL DEFAULT 'local',
    provider_id TEXT,
    invited_by TEXT REFERENCES users(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(provider, provider_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    jwt_token TEXT NOT NULL,
    expires_at DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS invite_tokens (
    token TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    role TEXT NOT NULL,
    created_by TEXT NOT NULL REFERENCES users(id),
    expires_at DATETIME NOT NULL,
    used_at DATETIME
);
CREATE TABLE IF NOT EXISTS connector_configs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL CHECK(type IN ('gdrive','s3','sharepoint','wikipedia','fda','edgar','espn')),
    name TEXT NOT NULL,
    credentials_json TEXT NOT NULL,
    schedule_cron TEXT,
    webhook_secret TEXT,
    created_by TEXT REFERENCES users(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS connector_uc_scopes (
    connector_id TEXT NOT NULL REFERENCES connector_configs(id) ON DELETE CASCADE,
    uc_id TEXT NOT NULL,
    scope_config_json TEXT NOT NULL,
    last_sync_at DATETIME,
    last_delta_token TEXT,
    PRIMARY KEY (connector_id, uc_id)
);
CREATE TABLE IF NOT EXISTS sync_runs (
    id TEXT PRIMARY KEY,
    connector_id TEXT NOT NULL REFERENCES connector_configs(id),
    uc_id TEXT NOT NULL,
    trigger TEXT NOT NULL CHECK(trigger IN ('manual','scheduled','webhook')),
    status TEXT NOT NULL CHECK(status IN ('running','ok','error')),
    files_total INTEGER DEFAULT 0,
    files_done INTEGER DEFAULT 0,
    error TEXT,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    finished_at DATETIME
);
