# KVForge Enterprise Ingestion — Design Spec
**Date:** 2026-05-07  
**Areas:** 1 (Enterprise Document Formats), 2 (Cloud Source Connectors), 3 (Incremental Sync & Change Detection)  
**Status:** Approved for implementation planning

---

## Goal

Make KVForge adoptable by enterprises using their existing documents, emails, spreadsheets, and images — by adding native support for Office formats, pluggable cloud source connectors (SharePoint, Google Drive, S3), and an incremental sync engine that keeps the vector corpus current without full re-indexing. Add PII detection, HITL content governance, and temporal chunk tracking as cross-cutting enterprise requirements.

---

## Key Architectural Principles

### 1. Self-hosted first, SaaS-extensible by design
All work targets self-hosted EC2/on-prem deployment. Three design decisions keep the SaaS path open without doing SaaS work now:
- **`CredentialStore` protocol** — local file/keychain backend in v1; cloud vault (AWS Secrets Manager, HashiCorp Vault) slots in later
- **`SyncScheduler` protocol** — in-process APScheduler backend in v1; distributed queue (Celery/SQS) slots in later
- **`tenant_id: str = "default"`** — present in all new data models from day one; no-op for self-hosted, becomes the isolation key for SaaS

### 2. Plugin pattern throughout
All new integrations follow KVForge's existing `@runtime_checkable` Protocol pattern. No ABC inheritance. New connectors, credential stores, content policies, and schedulers are added by implementing the protocol — zero changes to existing code.

### 3. Structural metadata as dual-purpose data
Document structure (heading level, slide number, sheet name, table position) is retained on every chunk. It serves two purposes: richer retrieval context for the model, and the granularity unit for incremental sync diffing.

### 4. Air-gapped support via local mirror path
Every cloud connector accepts an optional `local_mirror_path`. When set, KVForge reads from the local filesystem instead of calling the cloud API. Enterprises that cannot expose outbound cloud API calls mirror the source system to a local path using their own tooling; KVForge is agnostic.

### 5. Pure Python, cross-platform
All new libraries must work on Windows, macOS, and Linux without system dependencies. No LibreOffice subprocess, no COM automation, no platform-native bindings.

---

## Phase 1 — Foundation: Get Documents In (~4–5 weeks)

### 1a. New Document Format Loaders

**Priority order:** `.docx` → `.pptx` → `.xlsx` → ZIP archives (images already supported)

#### `.docx` loader (`ingestion/docx_loader.py`)
Library: `python-docx`

Structural metadata retained per chunk:
| Field | Description |
|---|---|
| `heading_level` | 1/2/3/None — heading style of the enclosing section |
| `heading_text` | Text of the nearest ancestor heading |
| `is_table` | True if chunk originates from a table cell |
| `table_position` | `{"row": int, "col": int}` for table-sourced chunks |
| `author` | `core_properties.author` from document metadata |
| `modified` | `core_properties.modified` ISO timestamp |
| `source_file` | Relative path within the source |

Chunking strategy: paragraph-boundary-aware chunking. Never split mid-paragraph. Tables are chunked row-by-row with column headers prepended to each row chunk.

**Section unit for diffing:** heading section (all paragraphs under one h1/h2/h3 heading). Hash = SHA-256 of concatenated paragraph text within the section.

#### `.pptx` loader (`ingestion/pptx_loader.py`)
Library: `python-pptx`

Structural metadata retained per chunk:
| Field | Description |
|---|---|
| `slide_number` | 1-based slide index |
| `slide_title` | Text of the title placeholder if present |
| `speaker_notes` | Full speaker notes text (indexed separately as a companion chunk) |
| `shape_type` | `"text"` / `"table"` / `"image"` |
| `shape_bbox` | `{"left": float, "top": float, "width": float, "height": float}` in EMU, normalised to 0–1 |

Chunking strategy: one chunk per slide (text + alt-text of images combined). Speaker notes become a companion chunk tagged `is_speaker_notes: true`. Tables chunked row-by-row.

**Section unit for diffing:** slide. Hash = SHA-256 of all text content + speaker notes on the slide.

#### `.xlsx` loader (`ingestion/xlsx_loader.py`)
Library: `openpyxl`

Structural metadata retained per chunk:
| Field | Description |
|---|---|
| `sheet_name` | Name of the worksheet |
| `table_name` | Excel table name if the range is a defined table, else None |
| `row_range` | `{"start": int, "end": int}` 1-based row numbers for this chunk |
| `column_headers` | List of header strings prepended to each chunk |
| `named_range` | Named range name if the chunk falls within one |

Chunking strategy: header row detected automatically (first non-empty row). Subsequent rows chunked in windows of `chunk_size` rows (default: 50). Column headers prepended to every chunk so each chunk is self-contained.

**Section unit for diffing:** sheet. Hash = SHA-256 of all cell values in the sheet. For large sheets (>10K rows), hash is computed per named-table or per 500-row window — whichever is smaller.

#### ZIP archive loader (`ingestion/zip_loader.py`)
Unpacks to a temp directory, dispatches each file to the appropriate loader via `ingestion/registry.py`. Nested ZIPs supported to one level. Unsupported extensions skipped silently.

#### Directory loader update (`ingestion/directory_loader.py`)
Add to `EXTENSION_MAP`:
```python
".docx": "docx",
".pptx": "pptx",
".xlsx": "xlsx",
".zip":  "zip",
```

### 1b. SourceConnector Protocol (`connectors/base.py`)

```python
@runtime_checkable
class SourceConnector(Protocol):
    def list_files(self) -> list[SourceFile]: ...
    def download(self, file: SourceFile) -> bytes: ...
    def get_modified_at(self, file: SourceFile) -> datetime: ...
    def supports_delta(self) -> bool: ...         # True if connector supports delta tokens
    def get_delta(self, token: str | None) -> tuple[list[SourceFile], str]: ...
```

`SourceFile` dataclass:
```python
@dataclass
class SourceFile:
    id: str                    # connector-native unique ID (stable across renames)
    name: str                  # filename with extension
    path: str                  # full path within the source (for display)
    size: int                  # bytes
    modified_at: datetime
    mime_type: str | None = None
    extra: dict = field(default_factory=dict)  # connector-specific metadata
```

### 1c. S3 Connector (`connectors/s3_connector.py`)
Library: `boto3`

Config fields added to `DatasourceConfig`:
```python
s3_bucket: str = ""
s3_prefix: str = ""            # folder prefix to restrict indexing
s3_region: str = "us-east-1"
s3_access_key_id: str = ""
s3_secret_access_key: str = ""
local_mirror_path: str = ""    # if set, read from local filesystem instead of S3 API
```

`supports_delta()` returns False. Change detection via `modified_at` + file size comparison.

### 1d. CredentialStore Protocol (`connectors/credential_store.py`)

```python
@runtime_checkable
class CredentialStore(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    def delete(self, key: str) -> None: ...
```

Phase 1 ships `LocalFileCredentialStore` — encrypts to `~/.kvforge/credentials.enc` using a machine-derived key. Cloud vault backend (Phase 3+) implements the same protocol.

### 1e. PII Detection Pipeline (`core/pii_detector.py`)

Three-gate pipeline, runs before chunking:

**Gate 1 — Rule-based (structured PII)**  
Regex patterns for: SSN (`\b\d{3}-\d{2}-\d{4}\b`), credit card (Luhn-validated), IBAN, UK NI number, phone numbers (E.164 + local formats), email addresses.

**Gate 2 — NER-based (unstructured PII)**  
Model: spaCy `en_core_web_sm` (12MB, CPU, pure Python). Entity types: `PERSON`, `GPE` (addresses), `ORG` (when policy requires). Configurable per UC via `allowed_pii_categories`.

**Gate 3 — Fine-tuning exclusion**  
Any chunk that triggered PII detection (even after redaction) is tagged `pii_flagged: true`. The LoRA trainer and FAQ generator skip chunks with this flag.

**Redaction:** PII spans replaced with typed placeholders `[PERSON]`, `[SSN]`, `[EMAIL]`, `[PHONE]`, `[CREDIT_CARD]`. Sanitised text is stored; original never persisted anywhere.

**PII audit log:** Append-only SQLite table `pii_audit` — columns: `file_path`, `chunk_id`, `detected_category`, `action` (`redacted` / `rejected`), `timestamp`. PII values never logged.

**Config additions to `DatasourceConfig`:**
```python
pii_detection_enabled: bool = True
allowed_pii_categories: list[str] = Field(default_factory=list)
pii_rejection_threshold: int = 3   # reject whole chunk if N+ PII spans found after redaction attempts
```

### 1f. Chunk Timestamp Fields

Every chunk upserted to the vector store gains three new payload fields:
```python
effective_from: str    # ISO datetime — when this chunk version became active
superseded_at: str | None = None   # ISO datetime — when replaced; None = currently active
source_version: str    # source document's last-modified timestamp at index time
```

Default retrieval adds filter `superseded_at IS NULL` (or vector-store equivalent). Point-in-time queries supported by filtering `effective_from <= target_date AND (superseded_at IS NULL OR superseded_at > target_date)`.

### 1g. DatasourceConfig additions (Phase 1)
```python
tenant_id: str = "default"
loader: str = "directory"          # existing field — now also accepts "s3"
sync_interval_minutes: int = 60    # Area 3 — used from Phase 2
hitl_mode: Literal["blocking", "non-blocking", "auto"] = "auto"   # Area 3 / HITL
allowed_pii_categories: list[str] = Field(default_factory=list)
pii_detection_enabled: bool = True
pii_rejection_threshold: int = 3
local_mirror_path: str = ""
```

---

## Phase 2 — Connect & Sync: Live Sources (~5–6 weeks)

### 2a. SharePoint Connector (`connectors/sharepoint_connector.py`)
Library: `msal` (Microsoft Authentication Library, pure Python)

Two auth modes (both configured in the same connector, selected by `sharepoint_auth_mode`):

**Service principal** (background sync):
- Config: `sharepoint_tenant_id`, `sharepoint_client_id`, `sharepoint_client_secret`
- Acquires token via client credentials flow, no user interaction
- Permission required: `Sites.Read.All` on the Azure AD app registration

**Delegated OAuth2** (Studio setup wizard only):
- Launches browser-based auth flow from Studio
- Lists available sites and document libraries for the user to pick
- On selection, stores the site ID and library ID to `DatasourceConfig`
- Subsequent background syncs use service principal

`supports_delta()` returns True. Uses Microsoft Graph `$deltaLink` for efficient incremental polling. Falls back to `modified_at` comparison when delta token expires.

`local_mirror_path` supported: if set, reads from local filesystem instead of calling Graph API.

### 2b. Google Drive Connector (`connectors/gdrive_connector.py`)
Library: `google-auth`, `google-api-python-client` (pure Python)

Auth: OAuth2 service account JSON key file (for background sync) + user consent flow (for Studio wizard to browse shared drives and pick a folder).

`supports_delta()` returns True via Drive API change tokens.

`local_mirror_path` supported.

### 2c. Studio Connector Wizard

New Studio wizard tab: **Connect a Source**. Steps:
1. Pick connector type (SharePoint / Google Drive / S3 / Local Mirror)
2. Enter credentials (service account key, client ID/secret, or AWS credentials)
3. Browser-based auth flow for OAuth connectors — lists available sites/drives/buckets
4. User picks the specific library/folder/prefix to index
5. Set sync interval (slider: 5 min → 24 hours, default 60 min)
6. Review and launch first sync

### 2d. SyncScheduler Protocol (`core/sync_scheduler.py`)

```python
@runtime_checkable
class SyncScheduler(Protocol):
    def schedule(self, uc_name: str, interval_minutes: int, fn: Callable) -> str: ...
    def cancel(self, job_id: str) -> None: ...
    def list_jobs(self) -> list[SyncJob]: ...
    def trigger_now(self, job_id: str) -> None: ...
```

Phase 2 ships `APSchedulerBackend` — runs in-process using `APScheduler`. The `SyncJob` dataclass carries `uc_name`, `job_id`, `interval_minutes`, `last_run`, `next_run`, `last_status`.

### 2e. Section-Hash Diffing Engine (`core/sync_engine.py`)

**Sync state store:** SQLite DB `<collection>_sync.db` (separate from replay buffer). Schema:
```sql
CREATE TABLE section_hashes (
    uc_name       TEXT NOT NULL,
    source_id     TEXT NOT NULL,   -- connector-native file ID (stable across renames)
    section_id    TEXT NOT NULL,   -- "slide:3", "heading:Introduction", "sheet:Revenue"
    content_hash  TEXT NOT NULL,   -- SHA-256 of section text
    chunk_ids     TEXT NOT NULL,   -- JSON array of vector store chunk IDs for this section
    indexed_at    TEXT NOT NULL,
    PRIMARY KEY (uc_name, source_id, section_id)
);

CREATE TABLE document_hashes (
    uc_name       TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    doc_hash      TEXT NOT NULL,   -- SHA-256 of full document (fast pre-check)
    modified_at   TEXT NOT NULL,
    PRIMARY KEY (uc_name, source_id)
);

CREATE TABLE deleted_docs (
    uc_name       TEXT NOT NULL,
    source_id     TEXT NOT NULL,
    deleted_at    TEXT NOT NULL
);
```

**Sync algorithm per document:**
```
1. Fetch modified_at from source → compare with stored modified_at
   If unchanged → skip (no API call for content)
2. If changed → download document
3. Compute doc_hash (SHA-256 of raw bytes) → compare with stored doc_hash
   If unchanged → update modified_at only (e.g. metadata-only change)
4. If changed → extract sections and compute per-section hashes
5. For each section:
   a. If hash unchanged → skip (keep existing chunks, KV tensors, access tier intact)
   b. If hash changed → 
      - Set superseded_at = now on old chunk IDs for this section
      - Re-chunk + re-embed new section content
      - Upsert new chunks with effective_from = now
      - Update section_hashes record
6. Update document_hashes record
```

**Deletion detection:**
```
1. Get set of source_ids from last sync (from document_hashes table)
2. Get set of source_ids from current connector.list_files()
3. Deleted = last_sync_ids - current_ids
4. For each deleted source_id: set superseded_at = now on all its chunks, insert into deleted_docs
```

### 2f. Phase Regression Hook

After each sync run, the sync engine evaluates the change threshold:

**Option B (percentage):** `changed_chunks / total_uc_chunks > regression_pct_threshold`  
**Option C (access tier):** `hot_or_warm_chunks_changed / total_hot_warm_chunks > regression_tier_threshold`

Either condition firing triggers `append_prs()` with a synthetic PRS of 0.0 and `regression_threshold=1.0` — forcing an immediate Phase 3 → Phase 2 downgrade. Phase 3 is only restored after a new LoRA round + PRS re-evaluation passes the normal threshold.

Config additions:
```python
sync_regression_mode: Literal["pct", "tier", "either"] = "either"
sync_regression_pct_threshold: float = 0.10
sync_regression_tier_threshold: float = 0.15
```

### 2g. Stale FAQ Detection

`sleep_faq_generator.py` tags each generated FAQ with `source_chunk_ids: list[str]`.  

After each sync run, the sync engine queries: which FAQ `source_chunk_ids` are now `superseded_at IS NOT NULL`? Those FAQs are marked `stale: true` in a new `faq_metadata` SQLite table. The LoRA trainer skips stale FAQs; the FAQ generator re-generates them from the updated chunks on next sleep-time run.

### 2h. Temporal Grounding in FAQ Generation

`sleep_faq_generator.py` updated to include document `effective_from` date in the generation prompt context:

```
Context (as of {effective_from_date}):
{chunk_text}

Generate a FAQ question and answer based on the above context. 
The answer should reflect information current as of {effective_from_date}.
```

### 2i. Sync History Dashboard

New panel in Studio UC detail page: **Sync History**. Columns: timestamp, source, files checked, files changed, chunks superseded, new chunks added, PII detections, errors. Last 30 runs shown. Backed by a new `sync_runs` table in `<collection>_sync.db`.

---

## Phase 3 — Governance & Trust (~6–7 weeks)

### 3a. ContentPolicy Protocol (`core/content_policy.py`)

```python
@runtime_checkable
class ContentPolicy(Protocol):
    def evaluate(self, chunk: dict, existing_chunks: list[dict]) -> PolicyResult: ...
```

`PolicyResult` dataclass: `action: Literal["approve", "flag_conflict", "flag_domain", "reject"]`, `reason: str`, `confidence: float`, `conflicting_chunk_ids: list[str]`.

Two built-in implementations:
- `ConflictDetectionPolicy` — semantic similarity + lexical divergence
- `DomainRelevancePolicy` — embedding distance from UC centroid

### 3b. Conflict Detection (`core/content_policy.py`)

For each new chunk:
1. Retrieve top-5 most similar existing chunks (cosine similarity > 0.85)
2. For each similar chunk, compute lexical overlap (Jaccard on trigrams)
3. If cosine_sim > 0.85 AND jaccard < 0.25 → same topic, different facts → `flag_conflict`

Conflict pairs stored in `content_flags` table with both chunk IDs for the reviewer.

### 3c. Domain Relevance Filter (`core/content_policy.py`)

UC centroid computed as mean of all chunk embeddings. New chunk's cosine distance from centroid compared against `domain_relevance_threshold` (default: 0.60). Below threshold → `flag_domain`.

Centroid is updated incrementally after each sync run (running mean, no full recompute).

### 3d. Studio HITL Review Queue

New Studio page: **Content Review**. Shows all pending flags grouped by UC.

Per-flag actions:
| Action | Effect |
|---|---|
| **Approve** | `content_flags.resolved = true`, chunk eligible for fine-tuning |
| **Reject** | `superseded_at = now` on chunk, chunk removed from retrieval |
| **Supersede** | Resolver picks winning chunk; loser gets `superseded_at = now` |
| **Annotate** | Expert note stored as `reviewer_note` in chunk payload; note included in retrieval context |

**`hitl_mode` behaviour:**
- `blocking` — chunk held in `pending` state, not in vector store until approved
- `non-blocking` — chunk in vector store immediately, tagged `under_review: true`; shown with `[under review]` badge in citations; excluded from fine-tuning
- `auto` — UC type heuristic: UCs with `hitl_sensitivity: "high"` → blocking; others → non-blocking. `hitl_sensitivity` is a new `DatasourceConfig` field.

### 3e. Document ACL Metadata

Each chunk gains two new payload fields:
```python
acl_users: list[str] = []     # email addresses or user IDs permitted to see this chunk
acl_groups: list[str] = []    # group names permitted to see this chunk
```

Populated at index time from the source connector: SharePoint and Google Drive expose file-level permissions via their APIs. S3 tags can carry `x-amz-acl-users` / `x-amz-acl-groups` custom metadata.

Per-query user filtering: query API accepts optional `requesting_user: str` and `requesting_groups: list[str]`. If set, vector store query adds a filter: `acl_users contains user OR acl_groups intersects groups OR (acl_users is empty AND acl_groups is empty)` (empty ACL = public to all UC users).

### 3f. LDAP / Active Directory Group Sync (`core/ldap_sync.py`)

Scheduled job (runs before each sync cycle): fetches group membership from LDAP/AD, updates a local `groups` SQLite cache. The per-query group filter resolves group names against this cache. Config:
```python
ldap_host: str = ""
ldap_port: int = 389
ldap_bind_dn: str = ""
ldap_bind_password: str = ""
ldap_base_dn: str = ""
ldap_group_filter: str = "(objectClass=groupOfNames)"
```

### 3g. SSO / OAuth2 for Studio Login

Studio gains a login gate backed by a `StudioAuthProvider` protocol. Phase 3 ships an OAuth2 provider (Azure AD and Google Workspace). SAML provider follows in a future phase. Session tokens stored in HTTP-only cookies, 8-hour expiry.

### 3h. Query Audit Log

Every query to the monitoring dashboard `/api/query` endpoint is logged to `query_audit` SQLite table: `timestamp`, `uc_name`, `user_id`, `query_text`, `returned_chunk_ids` (JSON array), `answer_hash` (SHA-256 of answer — not the answer itself), `phase_used` (1/2/3), `latency_ms`.

Compliance export: new endpoint `GET /api/uc/{uc_name}/audit-export?from=&to=` returns a JSON-L file of all query audit records in the range. Suitable for GDPR/SOC2 review.

### 3i. SaaS-Readiness Completions

- `CredentialStore` cloud backend: `AWSSecretsManagerCredentialStore` + `HashiCorpVaultCredentialStore`
- `SyncScheduler` distributed backend: `CelerySchedulerBackend` (SQS as broker)
- Admin console page in Studio: org-wide view of all UCs, users, active sync jobs, usage (query count, GPU hours, storage GB)

---

## Module Map

**New files:**
```
ingestion/
├── docx_loader.py
├── pptx_loader.py
├── xlsx_loader.py
└── zip_loader.py

connectors/
├── __init__.py
├── base.py                  — SourceConnector protocol, SourceFile dataclass
├── credential_store.py      — CredentialStore protocol, LocalFileCredentialStore
├── s3_connector.py
├── sharepoint_connector.py
└── gdrive_connector.py

core/
├── pii_detector.py          — 3-gate PII pipeline
├── sync_engine.py           — section-hash diffing, deletion detection
├── sync_scheduler.py        — SyncScheduler protocol, APSchedulerBackend
├── content_policy.py        — ContentPolicy protocol, ConflictDetectionPolicy, DomainRelevancePolicy
└── ldap_sync.py

tests/
├── test_docx_loader.py
├── test_pptx_loader.py
├── test_xlsx_loader.py
├── test_zip_loader.py
├── test_s3_connector.py
├── test_sharepoint_connector.py
├── test_gdrive_connector.py
├── test_pii_detector.py
├── test_sync_engine.py
├── test_content_policy.py
└── test_ldap_sync.py
```

**Modified files:**
```
core/config.py               — new DatasourceConfig fields (all phases)
ingestion/directory_loader.py — add docx/pptx/xlsx/zip to EXTENSION_MAP
ingestion/registry.py        — register new loaders
pipeline/kv_indexer.py       — add effective_from/source_version to chunk upsert
pipeline/prs_evaluator.py    — skip pii_flagged chunks
pipeline/sleep_faq_generator.py — temporal grounding prompt, source_chunk_ids tagging
core/version.py              — sync-triggered phase regression hook
studio/api.py                — connector wizard endpoints, review queue endpoints
studio/routes.py             — new routes: /connector-wizard, /review-queue, /sync-history
templates/studio/
├── hub.html                 — sync status pills per UC
├── uc_detail.html           — sync history panel
└── review_queue.html        — new: HITL review queue page
```

---

## Data Model Summary

### New SQLite DB: `<collection>_sync.db`
Tables: `section_hashes`, `document_hashes`, `deleted_docs`, `sync_runs`, `faq_metadata`, `content_flags`, `query_audit`, `groups` (LDAP cache), `pii_audit`

### Vector store chunk payload additions
| Field | Phase | Type | Description |
|---|---|---|---|
| `effective_from` | 1 | ISO datetime string | When this chunk became active |
| `superseded_at` | 1 | ISO datetime string or null | When replaced |
| `source_version` | 1 | ISO datetime string | Source doc last-modified at index time |
| `structural_metadata` | 1 | JSON object | heading_level, slide_number, sheet_name, etc. |
| `pii_flagged` | 1 | bool | True if PII was detected in this chunk |
| `under_review` | 3 | bool | True if HITL flag is unresolved |
| `reviewer_note` | 3 | string or null | Expert annotation from HITL review |
| `acl_users` | 3 | JSON array of strings | Permitted user IDs |
| `acl_groups` | 3 | JSON array of strings | Permitted group names |

---

## Testing Approach

All tests are pure Python using `pytest`, no GPU, no live external services. Connectors are tested with mocked API responses (`unittest.mock`). The PII detector is tested against a fixed corpus of synthetic PII strings. The sync engine is tested against a `LocalFileConnector` (filesystem-backed test double). HITL and ACL logic tested via `fastapi.testclient.TestClient` with patched `ROOT`.

Each new module follows the existing TDD pattern: failing tests first, minimal implementation, passing tests, commit.

---

## Open Questions (deferred)

- Email formats (`.eml` / `.msg`): not in v1 scope; can be added as a Phase 2 loader following the same structural-metadata pattern (sender, subject, date, thread-id as metadata fields).
- CSV: `openpyxl` does not read CSV; add a separate `csv_loader.py` using stdlib `csv` module if needed.
- Webhook / push-based sync (Microsoft Graph change notifications): deferred to post-Phase 2 as an opt-in enhancement on top of the pull scheduler.
- SAML SSO provider: deferred to post-Phase 3.
- Target scale: design assumes corpora up to ~500K documents. The section-hash diffing SQLite DB and in-process APScheduler are suitable at this scale. Above 1M documents, evaluate migrating sync state to PostgreSQL and scheduler to distributed queue.
