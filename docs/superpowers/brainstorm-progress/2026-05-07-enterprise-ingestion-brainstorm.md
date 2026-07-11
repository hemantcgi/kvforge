# Enterprise Ingestion Brainstorm — Session Progress
**Date:** 2026-05-07  
**Areas in scope:** Area 1 (Enterprise Document Formats), Area 2 (Cloud Source Connectors), Area 3 (Incremental Sync & Change Detection)  
**Status:** In progress — Q8 not yet answered

---

## Decisions Locked

### Q1 — Deployment model
**Decision: A — Self-hosted**, with three design decisions that keep SaaS (B) viable later:
1. `CredentialStore` protocol — abstracts local file/keychain vs. cloud vault (AWS Secrets Manager etc.)
2. `SyncScheduler` protocol — abstracts in-process cron vs. distributed job queue (Celery/SQS)
3. `tenant_id` field in all data models from day one (`"default"` for self-hosted, real ID for SaaS)

---

### Q2 — Document format priority
**Decision: .docx → .pptx → .xlsx → jpeg/png (in order)**

Constraints:
- Pure Python libraries only — no LibreOffice subprocess, no COM automation, no platform-native bindings
- Cross-platform: Windows, macOS, Linux
- Libraries: `python-docx`, `python-pptx`, `openpyxl`, Pillow

Structural metadata to retain per format:
- `.docx` — heading level (h1/h2/h3), paragraph style, table structure, author, modified date
- `.pptx` — slide number, slide title, speaker notes, shape positions, image bounding boxes
- `.xlsx` — sheet name, row/column headers, named ranges, table ranges
- images — EXIF data, position context from parent document (pptx/docx)

Metadata serves double duty: richer retrieval context AND granularity unit for incremental sync diffing.

---

### Q3 — Cloud connector priority
**Decision: SharePoint/OneDrive first** via Microsoft Graph API  
**Framework: `SourceConnector` Protocol** (same plugin pattern as vector stores/embedders) — Google Drive, S3, Confluence, Gmail slot in later by implementing the same interface.

---

### Q4 — Sync strategy
**Decision: A — Pull/scheduled polling**  
- Polling interval stored in `DatasourceConfig` as `sync_interval_minutes: int = 60`
- Minimum: 5 minutes, configurable per UC from Studio
- No inbound ports required — works in any private network

---

### Q5 — Change detection & re-indexing
**Decision: Section-hash diffing using document's natural structural boundaries**

How it works:
1. Compute whole-document hash first — if unchanged, skip entirely
2. If changed, compare per-section hashes:
   - `.pptx` → slide is the unit
   - `.docx` → heading section is the unit
   - `.xlsx` → sheet / named-table row-range is the unit
   - standalone images → whole file hash
3. Only re-chunk + re-embed sections whose hash changed
4. Unchanged sections keep their KV tensors, access tier history, training weight

Avoids chunk-boundary-shift problem of naive chunk-level diffing. Structural metadata (already retained for Q2) doubles as the diffing anchor.

---

### Q6 — SharePoint authentication
**Decision: C — Both service principal + delegated OAuth2**
- **Service principal** (client_id + client_secret, Azure AD app registration) — runs automated background sync unattended
- **Delegated OAuth2 flow** — used in Studio setup wizard so non-technical users can browse and pick SharePoint sites/libraries without needing to know site IDs

---

### Q7 — Phase regression threshold on document changes
**Decision: B or C — percentage threshold OR access-tier significance (whichever fires first)**
- B: configurable percentage of UC's total chunks superseded in one sync cycle (e.g. >10%)
- C: change concentrated in hot/warm-tier chunks (heavily accessed)
- Both thresholds configurable in `DatasourceConfig`
- NOT D (compound) — keep trigger logic simple, single condition

---

### Chronology architecture (emerged from discussion, not a numbered Q)
**In the Vector Database:**
- Every chunk gets: `effective_from`, `superseded_at` (null = active), `source_version` (doc last-modified)
- Default retrieval filters to `superseded_at IS NULL`
- Point-in-time queries supported ("as of Jan 2026") for compliance

**In the Fine-tuned LoRA model:**
- Temporal grounding in FAQ generation prompts ("As of [date], the policy states...")
- Sync-triggered phase regression: significant change → Phase 3 → Phase 2 (retrieval takes over)
- Stale FAQ detection: FAQs track `generated_from_chunk_ids`; when those chunks are superseded, FAQs are marked stale and excluded from next LoRA training batch

---

### PII handling (emerged from discussion)
**Three-gate defense in depth:**
1. **Pre-chunk scan** — rule-based regex (SSN, credit card, IBAN, phone, email) + NER (spaCy `en_core_web_sm`, 12MB CPU) for names/addresses
2. **Chunk-level redaction** — PII spans replaced with typed placeholders: `[PERSON]`, `[SSN]`, `[EMAIL]`. Sanitised text indexed, original never stored.
3. **Fine-tuning exclusion** — any chunk that triggered PII detection excluded from FAQ generation and LoRA training batches

Configurable policy per UC via `allowed_pii_categories` in `DatasourceConfig` (e.g. HR UC allows `["PERSON"]`).  
PII audit log: records file path, category, chunk position, action taken. Never logs actual PII values.

---

### HITL — Content verification (emerged from discussion)
**Conflict detection:** semantic similarity + lexical divergence between new chunks and existing. High cosine similarity + low token overlap = same topic, different facts = conflict flag.  
**Domain relevance:** embedding distance from UC centroid catches off-domain documents.

**Studio review queue** with four expert actions:
- **Approve** → normal indexing + eligible for fine-tuning
- **Reject** → removed from vector store, never trained on
- **Supersede** → expert resolves which version wins, old chunk gets `superseded_at = now`
- **Annotate** → correction note added to chunk metadata

Flagged chunks marked `[under review]` in answer citations. Excluded from LoRA training until expert acts.  
All governed by `ContentPolicy` protocol (same plugin pattern).

---

## Open Question — Next to Answer

### Q8 — HITL workflow timing
Should the review queue be blocking or non-blocking for indexing?

- **A** — Blocking (pre-index): flagged docs wait in queue, not indexed until expert approves. Max data quality, sync lag tied to review backlog.
- **B** — Non-blocking (post-index): flagged docs indexed immediately, marked `[under review]` in citations. Expert review gates fine-tuning only.
- **C** — Configurable per UC: sensitive UCs (legal, HR, finance) use blocking; general UCs (product docs, wikis) use non-blocking.

*Recommended: C*

---

## Questions Still to Ask
- Q9: Scope of Area 2 for phase 1 — how many connectors in v1?
- Q10: Offline / air-gapped enterprise support (affects connector design)?
- Q11: Phase breakdown — which features land in Phase 1 vs 2 vs 3?
- Q12: Target document volume and sync frequency at scale?

---

## Visual Companion
Server was running at http://localhost:55867 during this session. Feature landscape saved at:
`.superpowers/brainstorm/44801-1778164694/content/enterprise-landscape.html`
