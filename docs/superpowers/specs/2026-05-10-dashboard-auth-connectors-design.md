# KVForge Studio — Auth, Connector Management & Sync Dashboard Design

**Date:** 2026-05-10
**Branch:** kvforge-demos

---

## Goal

Extend KVForge Studio with a production-ready authenticated dashboard covering four subsystems built in dependency order:

1. **Auth** — invite-only signup, email/password + Google/Microsoft/AWS Cognito OAuth, three roles, SAML-ready
2. **Connector Management** — global credential configuration (admin) + per-UC scope configuration (editor) for GDrive, S3, SharePoint, and local file loaders
3. **Sync Engine** — manual, scheduled (APScheduler), and webhook-triggered syncs with real-time SSE progress
4. **UI Test Suite** — Playwright Python, 21 tests, screenshot capture on failure, lives in `tests/ui/` outside Studio

---

## Architecture Overview

Single FastAPI process (port 8080). All new code lives in three new top-level packages (`auth/`, `sync/`, `db/`) plus extensions to the existing `connectors/` package. `studio/` remains UI-routing-only. One SQLite database at `~/.kvforge/studio.db` for all auth, connector, and sync state — separate from existing per-UC replay DBs.

```
auth/                    ← new package
  models.py              ← User, Role, Session, InviteToken, OAuthAccount
  routes.py              ← /auth/login, /auth/logout, /auth/signup, /auth/me + OAuth callbacks
  oauth.py               ← Authlib clients: Google, Microsoft (MSAL), AWS Cognito; SAML stub
  middleware.py          ← Starlette AuthMiddleware: validates httpOnly JWT on every request

connectors/              ← existing package, extended
  base.py                ← existing SourceConnector Protocol (unchanged)
  gdrive_connector.py    ← existing (unchanged)
  s3_connector.py        ← existing (unchanged)
  sharepoint_connector.py ← existing (unchanged)
  credential_store.py    ← existing (unchanged)
  registry.py            ← new: ConnectorConfig CRUD backed by db/store.py
  sync_engine.py         ← new: orchestrates list→diff→download→ingest→upsert per connector+UC
  routes.py              ← new: /connectors REST API; role-gated

sync/                    ← new package
  scheduler.py           ← APScheduler AsyncIOScheduler; loads schedules from DB on startup
  webhook.py             ← POST /webhooks/{connector_id}; HMAC-SHA256 signature validation
  progress.py            ← asyncio.Queue SSE bus; GET /sync/stream/{connector_id}

db/                      ← new package
  schema.sql             ← DDL for all new tables
  store.py               ← thin sqlite3 wrapper: connect(), execute(), fetchall(), migrate()

studio/                  ← existing, minimal changes
  routes.py              ← add GET /studio/connectors page route (editor+ only)
  api.py                 ← unchanged
  (all other files)      ← unchanged

templates/studio/        ← existing templates
  connectors.html        ← new: connector dashboard page
  auth/
    login.html           ← new: sign-in page
    signup.html          ← new: invite-token signup page
    admin_users.html     ← new: admin user management table

tests/ui/                ← standalone QA module, outside Studio
  conftest.py
  test_auth.py
  test_connectors.py
  test_monitoring.py

kvforge_portal.py        ← modified: mount auth router; add AuthMiddleware
```

---

## Section 1: Auth

### User Model

```python
# auth/models.py
@dataclass
class User:
    id: str           # UUID
    email: str
    hashed_pw: str | None   # None for pure-OAuth users
    role: Literal["admin", "editor", "viewer"]
    provider: Literal["local", "google", "microsoft", "aws", "saml"]
    provider_id: str | None
    invited_by: str | None  # user ID of inviting admin
    created_at: datetime
```

### Roles

| Role | Capabilities |
|------|-------------|
| **Admin** | Invite users, change roles, configure global connector credentials, trigger syncs, view all |
| **Editor** | Configure per-UC connector scopes, trigger syncs, view all — cannot touch global credentials or user management |
| **Viewer** | View connector status, sync history, monitoring dashboards — no write access |

First user to complete signup is automatically assigned Admin role (bootstraps the system without manual DB seeding).

### Auth Methods

- **Email + password:** bcrypt hashed (cost factor 12). Stored in `users.hashed_pw`.
- **Google OAuth:** Authlib `GoogleClient`. Callback at `/auth/callback/google`.
- **Microsoft OAuth:** MSAL `ConfidentialClientApplication`. Callback at `/auth/callback/microsoft`.
- **AWS Cognito:** Authlib OIDC client pointed at Cognito user pool. Callback at `/auth/callback/aws`.
- **SAML (future):** `auth/saml.py` stub defines `SAMLProvider.handle_callback(request) → User`. Role derived from SAML group attributes. No invite token required for SAML users.

### Invite Flow

1. Admin POSTs `/auth/invite` with `{email, role}` → server creates `invite_tokens` record (one-time token, 48h TTL) and returns the signup URL.
2. Invitee opens `/auth/signup?token=<token>` → form pre-fills email and role (read-only).
3. Invitee sets password (or clicks an OAuth button to link account instead).
4. On submit: token is consumed (marked `used_at`), user record created, JWT session cookie set.
5. OAuth users: if OAuth account email matches invite email, OIDC callback auto-completes signup without password step.

### Session Management

- JWT stored as **httpOnly, Secure, SameSite=Lax** cookie — not localStorage (XSS-safe).
- JWT payload: `{sub: user_id, role, exp}`. Signed with `KVFORGE_SECRET_KEY` env var (HS256).
- Session record in `sessions` table for server-side revocation (logout invalidates the DB record).
- `AuthMiddleware` validates cookie on every request, attaches `request.state.user`. Returns 401 JSON for `/api/*` routes, 302 redirect to `/auth/login` for page routes.

### Routes

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/auth/login` | — | Login page |
| POST | `/auth/login` | — | Submit email+password → set cookie |
| GET | `/auth/logout` | any | Clear cookie, invalidate session |
| GET | `/auth/me` | any | Return current user JSON |
| POST | `/auth/invite` | admin | Generate invite token |
| GET | `/auth/signup` | — | Signup page (requires valid token param) |
| POST | `/auth/signup` | — | Create account from invite token |
| GET | `/auth/callback/{provider}` | — | OAuth callback handler |
| GET | `/studio/admin/users` | admin | User management page |
| GET | `/studio/api/users` | admin | List users JSON |
| PUT | `/studio/api/users/{id}/role` | admin | Change role |

---

## Section 2: Connector Management

### Data Model

**`connector_configs` table:**
```
id TEXT PK, type TEXT (gdrive|s3|sharepoint), name TEXT,
credentials_json TEXT (Fernet-encrypted),
schedule_cron TEXT (e.g. "*/30 * * * *"),
webhook_secret TEXT,
created_by TEXT (user_id), created_at DATETIME
```

**`connector_uc_scopes` table:**
```
connector_id TEXT, uc_id TEXT,
scope_config_json TEXT (folder_id/prefix/site_url per connector type),
last_sync_at DATETIME, last_delta_token TEXT,
PRIMARY KEY (connector_id, uc_id)
```

### Global Credentials (Admin only)

Credentials are encrypted using `cryptography.fernet.Fernet` with key derived from `KVFORGE_SECRET_KEY`. They are:
- Accepted via POST body (JSON)
- Encrypted before writing to SQLite
- **Never returned in plaintext** in any API response — all GET responses return `"credentials": "●●●●●●"` sentinel
- Decrypted in-memory only inside `sync_engine.py` at sync time

### Test Connection

POST `/connectors/{id}/test` — decrypts credentials, instantiates the connector, fires a minimal live check:
- **GDrive:** `service.files().list(pageSize=1).execute()`
- **S3:** `boto3.client.head_bucket(Bucket=bucket_name)`
- **SharePoint:** `GET /sites/{site_url}` via Graph API

Returns `{ok: true, detail: "..."}` or `{ok: false, error: "..."}` within 10s timeout.

### Per-UC Scope (Editor+)

Each UC can be linked to any configured connector with a scope config:
- **GDrive:** `{folder_id: "...", include_shared_drives: bool}`
- **S3:** `{bucket: "...", prefix: "docs/"}`
- **SharePoint:** `{site_url: "...", library: "Documents"}`
- **File loaders:** `{watch_dir: "~/.kvforge/watch/<uc_id>/", enabled_types: ["pdf","md","jsonl"]}`

### API Routes

All connector API routes mount under `/studio/api/connectors` via `connectors/routes.py`, consistent with the existing `api_router` prefix pattern in `studio/api.py`.

| Method | Path | Role | Purpose |
|--------|------|------|---------|
| GET | `/studio/api/connectors` | editor+ | List all configured connectors with status |
| POST | `/studio/api/connectors` | admin | Create connector (set global credentials) |
| PUT | `/studio/api/connectors/{id}` | admin | Update credentials or schedule |
| DELETE | `/studio/api/connectors/{id}` | admin | Remove connector and all scopes |
| POST | `/studio/api/connectors/{id}/test` | admin | Test connection live |
| GET | `/studio/api/connectors/{id}/scopes` | editor+ | List UC scopes for this connector |
| POST | `/studio/api/connectors/{id}/scopes` | editor | Add/update UC scope |
| DELETE | `/studio/api/connectors/{id}/scopes/{uc_id}` | editor | Remove UC scope |
| POST | `/studio/api/connectors/{id}/sync` | editor+ | Trigger manual sync (all scopes) |
| GET | `/webhooks/{connector_id}` | — (HMAC-validated) | Incoming push webhook |
| GET | `/sync/stream/{connector_id}` | editor+ | SSE progress stream |
| GET | `/studio/connectors` | editor+ | Connector dashboard page |

---

## Section 3: Sync Engine

### SyncEngine (`connectors/sync_engine.py`)

```python
class SyncEngine:
    def run(self, connector_id: str, uc_id: str, trigger: str) -> None:
        # 1. Load connector config, decrypt credentials
        # 2. Instantiate SourceConnector
        # 3. Emit progress: {stage: "discover", files_total: 0, files_done: 0}
        # 4. If supports_delta(): files, token = connector.get_delta(last_token)
        #    Else: files = connector.list_files(); diff against sync_state
        # 5. For each changed file:
        #    a. bytes = connector.download(file)
        #    b. loader = pick_loader(file.mime_type)
        #    c. chunks = loader.load(bytes)
        #    d. embed chunks (FastEmbed)
        #    e. compute KV tensors (LLM forward pass)
        #    f. upsert to vector DB
        #    g. Emit progress: {files_done: n}
        # 6. Persist delta token, update last_sync_at
        # 7. Write sync_runs record with final status
```

`SyncEngine.run()` is always called in a background thread (via `asyncio.run_in_executor`) so it never blocks the FastAPI event loop.

### Scheduler (`sync/scheduler.py`)

- `APScheduler AsyncIOScheduler` started in FastAPI `lifespan` context.
- On startup: reads all `connector_configs` with non-null `schedule_cron` and registers cron jobs.
- Each job calls `SyncEngine.run(connector_id, uc_id, trigger="scheduled")` for every scope of that connector.
- Admin can update schedule via `PUT /connectors/{id}` — scheduler removes old job and registers new one.

### Webhook Receiver (`sync/webhook.py`)

- `POST /webhooks/{connector_id}` — open endpoint (no auth cookie required).
- Validates `X-Hub-Signature-256` HMAC-SHA256 against `connector_configs.webhook_secret`.
- Returns 200 immediately, enqueues `SyncEngine.run(..., trigger="webhook")` as background task.
- Supported push sources:
  - **Google Drive:** Drive API push channels (change notifications)
  - **SharePoint:** Microsoft Graph change subscriptions
  - **S3:** S3 Event Notifications via SNS → HTTP endpoint

### SSE Progress (`sync/progress.py`)

Per-connector `asyncio.Queue`. `SyncEngine` publishes events as dicts:

```python
{
  "event": "progress",        # or "complete" | "error"
  "connector_id": "...",
  "uc_id": "...",
  "stage": "index",           # discover | index | kv_tensors
  "files_total": 847,
  "files_done": 312,
  "message": "embedding batch 3/9"
}
```

`GET /sync/stream/{connector_id}` drains the queue and yields `text/event-stream` lines. Browser EventSource reconnects automatically on disconnect. Same pattern as existing `pipeline_runner.py` SSE streaming in Studio.

### Sync Run History

Every sync writes a `sync_runs` record on start (status=`running`) and updates on completion (status=`ok`|`error`). Persisted to SQLite — survives server restarts. Displayed in the sync history table in the connector dashboard.

---

## Section 4: Sync Progress & Monitoring UI

### Connector Dashboard (`/studio/connectors`)

Three-panel layout:

**Panel 1 — Connector List:** Each configured connector shown as a card with:
- Name, type badge, health status dot (green/orange/red)
- Last sync time, files indexed count
- "Sync Now" button (editor+), "Configure" button (admin)

**Panel 2 — Active Sync Progress:** Visible when a sync is running. Shows:
- 3-stage progress bars: Discover files → Index & embed → Compute KV tensors
- File counts: discovered / changed / unchanged (skipped)
- Estimated time remaining
- Live log stream (SSE) showing current file being processed
- Trigger badge: manual | scheduled | webhook

**Panel 3 — Sync History Table:** Last 50 runs across all connectors. Columns: connector name, trigger, result (ok/error/running), new files, duration. Clicking an error row expands to show the error message.

### Monitoring Integration

The per-UC phase/PRS monitoring panel from `monitoring_dashboard.py` is embedded directly in the UC detail page in Studio (`/studio/uc/{uc_id}`). The separate per-UC monitoring ports (8081–8084) are no longer needed. Monitoring data is read from the existing `version.json` and `<uc>_replay.db` files — no new data storage required.

Monitoring panel shows: current phase (1/2/3), PRS score with threshold marker, LoRA version, chunks indexed, tier distribution (hot/warm/cold/frozen percentages), KV tensor freshness percentage, PRS history sparkline.

---

## Section 5: UI Test Suite

Lives in `tests/ui/` — a standalone Playwright Python test module that treats KVForge Studio as a black box. Not part of Studio; has no presence in Studio's routes or templates.

### Setup

```bash
pip install playwright pytest-playwright
playwright install chromium
pytest tests/ui/ -v --screenshot=only-on-failure
```

### conftest.py Fixtures

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `app_server` | session | Starts FastAPI on a random port via subprocess; yields `base_url`; tears down after all tests |
| `admin_page` | function | Playwright Page logged in as admin |
| `editor_page` | function | Playwright Page logged in as editor |
| `viewer_page` | function | Playwright Page logged in as viewer |
| `mock_gdrive` | function | Monkeypatches `GDriveConnector.list_files` to return 3 fake `SourceFile` objects; `download` returns `b"fake content"` |
| `mock_s3` | function | Same for `S3Connector` |
| `mock_sharepoint` | function | Same for `SharePointConnector` |
| `screenshot_on_failure` | function, autouse | `pytest_runtest_makereport` hook; captures full-page PNG to `tests/screenshots/<test_name>.png` on any failure |

No real GDrive / S3 / SharePoint credentials required in CI.

### Test Files

**`test_auth.py` (9 tests):**
- `test_login_valid_credentials` — email+password, assert redirect to `/studio/`
- `test_login_invalid_password` — assert error message, no redirect
- `test_invite_flow` — admin generates invite → user completes signup → can log in
- `test_invite_token_single_use` — second use of token returns 400
- `test_logout_clears_session` — after logout, `/studio/` redirects to `/auth/login`
- `test_viewer_cannot_access_connectors` — GET `/studio/connectors` as viewer → 403 page
- `test_editor_cannot_configure_credentials` — credential input fields are `readonly` for editors
- `test_unauthenticated_api_returns_401` — raw fetch to `/studio/api/registry` without cookie
- `test_admin_user_management_page` — user table renders, invite button present

**`test_connectors.py` (8 tests):**
- `test_add_gdrive_connector` — fill form, save, connector appears in list
- `test_connection_test_button` — click Test Connection, mock returns ok, success text visible
- `test_connection_test_failure` — mock raises `ConnectionError`, error message shown (always screenshotted)
- `test_add_uc_scope` — editor adds folder scope for a UC, scope persisted
- `test_credentials_masked_in_ui` — after save, credential field shows `●●●` not plaintext
- `test_manual_sync_trigger` — click Sync Now, progress card appears with `running` state
- `test_sync_progress_sse` — SSE events update progress bars; assert `files_done` percentage after mock emits events
- `test_sync_history_persists` — history table shows completed run after sync finishes

**`test_monitoring.py` (4 tests):**
- `test_phase_card_renders` — UC detail shows phase, PRS score, LoRA version
- `test_tier_distribution_visible` — hot/warm/cold percentages displayed
- `test_prs_history_chart` — sparkline renders; bar count matches history entries
- `test_error_connector_highlighted` — failed connector shows orange badge (always screenshotted)

---

## Database Schema

```sql
-- db/schema.sql

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    hashed_pw TEXT,
    role TEXT NOT NULL CHECK(role IN ('admin','editor','viewer')),
    provider TEXT NOT NULL DEFAULT 'local',
    provider_id TEXT,
    invited_by TEXT REFERENCES users(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
    type TEXT NOT NULL CHECK(type IN ('gdrive','s3','sharepoint')),
    name TEXT NOT NULL,
    credentials_json TEXT NOT NULL,  -- Fernet-encrypted
    schedule_cron TEXT,              -- NULL = no scheduled sync
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
```

---

## Security Decisions

| Concern | Decision |
|---------|----------|
| Session storage | httpOnly JWT cookie — not localStorage (XSS-safe) |
| Password hashing | bcrypt cost 12 |
| Credential storage | Fernet (AES-128-CBC) encrypted; key from `KVFORGE_SECRET_KEY` env var |
| Credential exposure | Never returned in API responses after initial save |
| OAuth CSRF | State param validated on every OAuth callback |
| Webhook authenticity | HMAC-SHA256 signature on `X-Hub-Signature-256` header |
| Path traversal | Existing `_uc_path()` guard in `studio/api.py` covers all UC ID lookups |
| Role enforcement | All connector credential endpoints check `request.state.user.role == "admin"` before executing |

---

## Environment Variables Required

| Variable | Purpose |
|----------|---------|
| `KVFORGE_SECRET_KEY` | JWT signing + Fernet encryption key (min 32 bytes, base64url-encoded for Fernet) |
| `GOOGLE_CLIENT_ID` | Google OAuth app client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth app client secret |
| `MICROSOFT_CLIENT_ID` | Azure AD app client ID |
| `MICROSOFT_CLIENT_SECRET` | Azure AD app client secret |
| `MICROSOFT_TENANT_ID` | Azure AD tenant ID |
| `AWS_COGNITO_POOL_ID` | Cognito user pool ID |
| `AWS_COGNITO_CLIENT_ID` | Cognito app client ID |
| `AWS_COGNITO_REGION` | Cognito region (e.g. `us-east-1`) |

All OAuth env vars are optional — if unset, the corresponding sign-in button is hidden on the login page.

---

## What Is Not In Scope

- Password reset via email (requires SMTP setup — deferred to a later spec)
- Multi-tenant isolation (all users share one Studio instance)
- Connector types beyond GDrive, S3, and SharePoint (new connectors implement `SourceConnector` Protocol)
- SAML implementation (stub interface defined, implementation deferred)
- Mobile-responsive design for auth / connector pages
