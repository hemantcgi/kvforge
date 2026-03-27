# vLLM All Use Cases + Portal Phase Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add vLLM inference to UC1/UC2/UC3, display phase status on the main portal, and push everything to GitHub.

**Architecture:** UC4's vLLM pattern (config fields `vllm_url`/`vllm_model` + `start_vllm.sh`) is replicated for UC1-3 with dynamic checkpoint detection from `version.json`. The portal's `/api/status` is enriched to fetch `/api/version` from each dashboard, and the portal cards display live phase badges. Deployment scripts are updated to start each UC's vLLM server after its pipeline completes.

**Tech Stack:** bash, Python, FastAPI, httpx, vLLM 0.18.0, PEFT LoRA, EC2 g5.xlarge (4× A10G), Qdrant, git

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `examples/usecase1_customer_support/config.json` | Modify | Add `vllm_url`, `vllm_model`, `quantization` |
| `examples/usecase2_pubmedqa/config.json` | Modify | Add `vllm_url`, `vllm_model`, `quantization` |
| `examples/usecase3_squad/config.json` | Modify | Add `vllm_url`, `vllm_model`, `quantization` |
| `examples/usecase1_customer_support/start_vllm.sh` | Create | GPU 0, port 8091, auto-detects checkpoint |
| `examples/usecase2_pubmedqa/start_vllm.sh` | Create | GPU 1, port 8092, auto-detects checkpoint |
| `examples/usecase3_squad/start_vllm.sh` | Create | GPU 2, port 8093, auto-detects checkpoint |
| `kvforge_portal.py` | Modify | `/api/status` fetches `/api/version`; cards show phase badge |
| `deploy_and_run.sh` | Modify | Start vLLM servers for UC1/UC2/UC3 after staggered pipeline launches |
| `run_all_pipelines.sh` | Modify | Start vLLM servers after each pipeline completes |

---

## Task 1: Add vLLM config fields to UC1, UC2, UC3

**Files:**
- Modify: `examples/usecase1_customer_support/config.json`
- Modify: `examples/usecase2_pubmedqa/config.json`
- Modify: `examples/usecase3_squad/config.json`

- [ ] **Step 1: Add vLLM fields to UC1 config**

In `examples/usecase1_customer_support/config.json`, after `"llm_model": "meta-llama/Llama-3.2-3B-Instruct"`, add:
```json
  "llm_model":        "meta-llama/Llama-3.2-3B-Instruct",
  "quantization":     "4bit",
  "vllm_url":         "http://localhost:8091",
  "vllm_model":       "uc1",
```

- [ ] **Step 2: Add vLLM fields to UC2 config**

In `examples/usecase2_pubmedqa/config.json`, after `"llm_model": "meta-llama/Llama-3.2-3B-Instruct"`, add:
```json
  "llm_model":        "meta-llama/Llama-3.2-3B-Instruct",
  "quantization":     "4bit",
  "vllm_url":         "http://localhost:8092",
  "vllm_model":       "uc2",
```

- [ ] **Step 3: Add vLLM fields to UC3 config**

In `examples/usecase3_squad/config.json`, after `"llm_model": "meta-llama/Llama-3.2-3B-Instruct"`, add:
```json
  "llm_model":        "meta-llama/Llama-3.2-3B-Instruct",
  "quantization":     "4bit",
  "vllm_url":         "http://localhost:8093",
  "vllm_model":       "uc3",
```

- [ ] **Step 4: Verify — validate all three configs parse cleanly**

Run from repo root:
```bash
python3 -c "
import json
for uc in ['usecase1_customer_support','usecase2_pubmedqa','usecase3_squad']:
    cfg = json.load(open(f'examples/{uc}/config.json'))
    assert 'vllm_url' in cfg, f'{uc} missing vllm_url'
    assert 'vllm_model' in cfg, f'{uc} missing vllm_model'
    assert cfg['quantization'] == '4bit', f'{uc} wrong quantization'
    print(f'{uc}: vllm_url={cfg[\"vllm_url\"]} vllm_model={cfg[\"vllm_model\"]}')
print('All configs OK')
"
```
Expected output:
```
usecase1_customer_support: vllm_url=http://localhost:8091 vllm_model=uc1
usecase2_pubmedqa: vllm_url=http://localhost:8092 vllm_model=uc2
usecase3_squad: vllm_url=http://localhost:8093 vllm_model=uc3
All configs OK
```

- [ ] **Step 5: Commit**
```bash
git add examples/usecase1_customer_support/config.json \
        examples/usecase2_pubmedqa/config.json \
        examples/usecase3_squad/config.json
git commit -m "feat: add vllm_url/vllm_model/quantization to UC1/UC2/UC3 configs"
```

---

## Task 2: Create start_vllm.sh for UC1, UC2, UC3

**Files:**
- Create: `examples/usecase1_customer_support/start_vllm.sh`
- Create: `examples/usecase2_pubmedqa/start_vllm.sh`
- Create: `examples/usecase3_squad/start_vllm.sh`

These scripts auto-detect the latest LoRA checkpoint from `version.json` (falling back to scanning `lora_checkpoints/`) so they work regardless of which training version was last completed. Pattern is identical to UC4's `start_vllm.sh`.

- [ ] **Step 1: Create start_vllm.sh for UC1**

Create `examples/usecase1_customer_support/start_vllm.sh`:
```bash
#!/usr/bin/env bash
# Start vLLM inference server for UC1 on GPU 0 (port 8091).
#
# Reads the latest LoRA checkpoint path from version.json so it works
# regardless of which training round last completed.
#
# Usage:
#   cd ~/kvforge && bash examples/usecase1_customer_support/start_vllm.sh
#
# Logs: examples/usecase1_customer_support/vllm.log

set -e

UC_DIR="examples/usecase1_customer_support"
CONFIG="$UC_DIR/config.json"
VERSION_FILE="$UC_DIR/version.json"
LOG="$UC_DIR/vllm.log"
PORT=8091
GPU=0
MODEL="meta-llama/Llama-3.2-3B-Instruct"
LORA_NAME="uc1"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found at $CONFIG"; exit 1
fi

# Resolve LoRA checkpoint: prefer version.json, fall back to latest versioned dir
if [ -f "$VERSION_FILE" ]; then
  LORA_DIR=$(python3 -c "import json; v=json.load(open('$VERSION_FILE')); print(v.get('checkpoint_path',''))" 2>/dev/null || true)
fi

if [ -z "$LORA_DIR" ] || [ ! -d "$LORA_DIR" ]; then
  # Scan for latest vN directory
  LORA_DIR=$(ls -d "$UC_DIR/lora_checkpoints/v"* 2>/dev/null | sort -V | tail -1 || true)
fi

if [ -z "$LORA_DIR" ] || [ ! -d "$LORA_DIR" ]; then
  echo "ERROR: No LoRA checkpoint found — run the pipeline first"
  echo "  Checked version.json and $UC_DIR/lora_checkpoints/v*"
  exit 1
fi

command -v python3 &>/dev/null || { echo "ERROR: python3 not found"; exit 1; }
python3 -c "import vllm" 2>/dev/null || { echo "ERROR: vllm not installed (pip install vllm)"; exit 1; }

echo "[UC1 vLLM] starting on GPU $GPU, port $PORT, model=$MODEL, lora=$LORA_NAME" | tee -a "$LOG"
echo "[UC1 vLLM] checkpoint: $LORA_DIR" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --enable-lora \
  --lora-modules "${LORA_NAME}=${LORA_DIR}" \
  --max-lora-rank 16 \
  --port "$PORT" \
  --host "0.0.0.0" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --dtype float16 \
  2>&1 | tee -a "$LOG"
```

Make it executable:
```bash
chmod +x examples/usecase1_customer_support/start_vllm.sh
```

- [ ] **Step 2: Create start_vllm.sh for UC2**

Create `examples/usecase2_pubmedqa/start_vllm.sh` — identical structure, change:
- `UC_DIR="examples/usecase2_pubmedqa"`
- `PORT=8092`
- `GPU=1`
- `LORA_NAME="uc2"`
- All log prefixes `[UC1 vLLM]` → `[UC2 vLLM]`

```bash
#!/usr/bin/env bash
# Start vLLM inference server for UC2 on GPU 1 (port 8092).
#
# Reads the latest LoRA checkpoint path from version.json so it works
# regardless of which training round last completed.
#
# Usage:
#   cd ~/kvforge && bash examples/usecase2_pubmedqa/start_vllm.sh
#
# Logs: examples/usecase2_pubmedqa/vllm.log

set -e

UC_DIR="examples/usecase2_pubmedqa"
CONFIG="$UC_DIR/config.json"
VERSION_FILE="$UC_DIR/version.json"
LOG="$UC_DIR/vllm.log"
PORT=8092
GPU=1
MODEL="meta-llama/Llama-3.2-3B-Instruct"
LORA_NAME="uc2"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found at $CONFIG"; exit 1
fi

if [ -f "$VERSION_FILE" ]; then
  LORA_DIR=$(python3 -c "import json; v=json.load(open('$VERSION_FILE')); print(v.get('checkpoint_path',''))" 2>/dev/null || true)
fi

if [ -z "$LORA_DIR" ] || [ ! -d "$LORA_DIR" ]; then
  LORA_DIR=$(ls -d "$UC_DIR/lora_checkpoints/v"* 2>/dev/null | sort -V | tail -1 || true)
fi

if [ -z "$LORA_DIR" ] || [ ! -d "$LORA_DIR" ]; then
  echo "ERROR: No LoRA checkpoint found — run the pipeline first"
  echo "  Checked version.json and $UC_DIR/lora_checkpoints/v*"
  exit 1
fi

command -v python3 &>/dev/null || { echo "ERROR: python3 not found"; exit 1; }
python3 -c "import vllm" 2>/dev/null || { echo "ERROR: vllm not installed (pip install vllm)"; exit 1; }

echo "[UC2 vLLM] starting on GPU $GPU, port $PORT, model=$MODEL, lora=$LORA_NAME" | tee -a "$LOG"
echo "[UC2 vLLM] checkpoint: $LORA_DIR" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --enable-lora \
  --lora-modules "${LORA_NAME}=${LORA_DIR}" \
  --max-lora-rank 16 \
  --port "$PORT" \
  --host "0.0.0.0" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --dtype float16 \
  2>&1 | tee -a "$LOG"
```

Make it executable:
```bash
chmod +x examples/usecase2_pubmedqa/start_vllm.sh
```

- [ ] **Step 3: Create start_vllm.sh for UC3**

Create `examples/usecase3_squad/start_vllm.sh` — identical structure, change:
- `UC_DIR="examples/usecase3_squad"`
- `PORT=8093`
- `GPU=2`
- `LORA_NAME="uc3"`
- All log prefixes → `[UC3 vLLM]`

```bash
#!/usr/bin/env bash
# Start vLLM inference server for UC3 on GPU 2 (port 8093).
#
# Reads the latest LoRA checkpoint path from version.json so it works
# regardless of which training round last completed.
#
# Usage:
#   cd ~/kvforge && bash examples/usecase3_squad/start_vllm.sh
#
# Logs: examples/usecase3_squad/vllm.log

set -e

UC_DIR="examples/usecase3_squad"
CONFIG="$UC_DIR/config.json"
VERSION_FILE="$UC_DIR/version.json"
LOG="$UC_DIR/vllm.log"
PORT=8093
GPU=2
MODEL="meta-llama/Llama-3.2-3B-Instruct"
LORA_NAME="uc3"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: config not found at $CONFIG"; exit 1
fi

if [ -f "$VERSION_FILE" ]; then
  LORA_DIR=$(python3 -c "import json; v=json.load(open('$VERSION_FILE')); print(v.get('checkpoint_path',''))" 2>/dev/null || true)
fi

if [ -z "$LORA_DIR" ] || [ ! -d "$LORA_DIR" ]; then
  LORA_DIR=$(ls -d "$UC_DIR/lora_checkpoints/v"* 2>/dev/null | sort -V | tail -1 || true)
fi

if [ -z "$LORA_DIR" ] || [ ! -d "$LORA_DIR" ]; then
  echo "ERROR: No LoRA checkpoint found — run the pipeline first"
  echo "  Checked version.json and $UC_DIR/lora_checkpoints/v*"
  exit 1
fi

command -v python3 &>/dev/null || { echo "ERROR: python3 not found"; exit 1; }
python3 -c "import vllm" 2>/dev/null || { echo "ERROR: vllm not installed (pip install vllm)"; exit 1; }

echo "[UC3 vLLM] starting on GPU $GPU, port $PORT, model=$MODEL, lora=$LORA_NAME" | tee -a "$LOG"
echo "[UC3 vLLM] checkpoint: $LORA_DIR" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --enable-lora \
  --lora-modules "${LORA_NAME}=${LORA_DIR}" \
  --max-lora-rank 16 \
  --port "$PORT" \
  --host "0.0.0.0" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --dtype float16 \
  2>&1 | tee -a "$LOG"
```

Make it executable:
```bash
chmod +x examples/usecase3_squad/start_vllm.sh
```

- [ ] **Step 4: Fix UC4's start_vllm.sh to use python3 for consistency**

The existing `examples/usecase4_bedrock_userguide/start_vllm.sh` uses bare `python` (which is absent on EC2). Update lines 35-40 in that file:

**Old (lines 35-40 in UC4 start_vllm.sh):**
```bash
command -v python &>/dev/null || { echo "ERROR: python not found"; exit 1; }
python -c "import vllm" 2>/dev/null || { echo "ERROR: vllm not installed (pip install vllm)"; exit 1; }
...
CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
```

**New:**
```bash
command -v python3 &>/dev/null || { echo "ERROR: python3 not found"; exit 1; }
python3 -c "import vllm" 2>/dev/null || { echo "ERROR: vllm not installed (pip install vllm)"; exit 1; }
...
CUDA_VISIBLE_DEVICES=$GPU python3 -m vllm.entrypoints.openai.api_server \
```

- [ ] **Step 5: Verify all scripts use python3 and have correct GPU/port assignments**
```bash
ls -la examples/usecase{1,2,3,4}_*/start_vllm.sh
# Check GPU/port assignments are unique across all 4
grep -h "^PORT=\|^GPU=" examples/usecase{1,2,3}_*/start_vllm.sh examples/usecase4_bedrock_userguide/start_vllm.sh
# Verify no bare 'python ' (without '3') remains in any start_vllm.sh
grep -h "python " examples/usecase{1,2,3,4}_*/start_vllm.sh | grep -v "python3" && echo "FAIL: bare python found" || echo "OK: all use python3"
```
Expected:
```
PORT=8091 / GPU=0   (UC1)
PORT=8092 / GPU=1   (UC2)
PORT=8093 / GPU=2   (UC3)
PORT=8090 / GPU=3   (UC4)
OK: all use python3
```

- [ ] **Step 6: Commit**
```bash
git add examples/usecase1_customer_support/start_vllm.sh \
        examples/usecase2_pubmedqa/start_vllm.sh \
        examples/usecase3_squad/start_vllm.sh \
        examples/usecase4_bedrock_userguide/start_vllm.sh
git commit -m "feat: add start_vllm.sh for UC1/UC2/UC3; fix UC4 to use python3"
```

---

## Task 3: Update portal to show phase badges per use case

**Files:**
- Modify: `kvforge_portal.py`

The portal's `/api/status` currently only checks `/api/health`. We enrich it to also call `/api/version` (which returns `{phase: 1|2|3, ...}`) and return `{uc1: {status: "online", phase: 2}, ...}`. The card HTML adds a `<span class="phase-badge">` that shows Phase 1 / Phase 2 / Phase 3.

- [ ] **Step 1: Update `/api/status` to include phase**

In `kvforge_portal.py`, replace the `get_status()` function (currently ~10 lines after `@app.get("/api/status")`):

**Old:**
```python
@app.get("/api/status")
async def get_status():
    """Check which use-case dashboards are reachable."""
    results = {}
    async with httpx.AsyncClient(timeout=2.0) as client:
        for uc in USE_CASES:
            try:
                r = await client.get(f"http://localhost:{uc['port']}/api/health")
                results[uc["id"]] = "online" if r.status_code == 200 else "error"
            except Exception:
                results[uc["id"]] = "offline"
    return results
```

**New:**
```python
@app.get("/api/status")
async def get_status():
    """Check which use-case dashboards are reachable and return their phase."""
    results = {}
    async with httpx.AsyncClient(timeout=2.0) as client:
        for uc in USE_CASES:
            base = f"http://localhost:{uc['port']}"
            try:
                r = await client.get(f"{base}/api/health")
                if r.status_code != 200:
                    results[uc["id"]] = {"status": "error", "phase": None}
                    continue
                try:
                    rv = await client.get(f"{base}/api/version")
                    phase = rv.json().get("phase") if rv.status_code == 200 else None
                except Exception:
                    phase = None
                results[uc["id"]] = {"status": "online", "phase": phase}
            except Exception:
                results[uc["id"]] = {"status": "offline", "phase": None}
    return results
```

- [ ] **Step 2: Add phase badge CSS to PORTAL_HTML**

Inside the `<style>` block of `PORTAL_HTML`, add after the `.status-dot.error` rule.

**Important:** The existing style block already defines `.p1`, `.p2`, `.p3` classes (used for the architecture explanation section). Do NOT replace or remove those rules. Add the new `.phase-badge` class alongside them, after `.status-dot.error`:

```css
  .phase-badge {
    display: inline-block;
    padding: 2px 7px;
    border-radius: 3px;
    font-size: 0.72em;
    font-weight: bold;
    letter-spacing: 0.5px;
    margin-left: 6px;
    vertical-align: middle;
  }
  .phase-badge.p1 { background: #1f3a5f; color: #7ab8ff; }
  .phase-badge.p2 { background: #1f4a2f; color: #7aff9e; }
  .phase-badge.p3 { background: #3a1f4a; color: #c97aff; }
  .phase-badge.unknown { background: #2a2a2a; color: #666; }
```

These `.phase-badge.p1/p2/p3` rules are scoped to `.phase-badge` elements and will not affect the existing `.phase.p1/p2/p3` elements in the architecture section.

- [ ] **Step 3: Add phase badge span to each card in PORTAL_HTML**

**Important:** The four dashboard cards are generated from the `USE_CASES` list via a Python f-string loop — they are **not** four separate hardcoded HTML blocks. There is **one** card template in the HTML string with `{uc['id']}` and `{uc['title']}` placeholders. You only need to make **one replacement** in the template.

Find the card template's `card-header` div inside the Python f-string (it contains `{uc['id']}` as part of the `id="dot-..."` attribute):

```html
    <div class="card-header">
      <span class="card-title">{uc['title']}</span>
      <span class="status-dot" id="dot-{uc['id']}"></span>
    </div>
```

Replace it with (one replacement only):
```html
    <div class="card-header">
      <span class="card-title">{uc['title']}</span>
      <div style="display:flex;align-items:center;gap:6px;">
        <span class="phase-badge unknown" id="phase-{uc['id']}">…</span>
        <span class="status-dot" id="dot-{uc['id']}"></span>
      </div>
    </div>
```

After saving, verify all 4 IDs are generated correctly by checking the rendered HTML at `http://localhost:8080/` — you should see `id="phase-uc1"`, `id="phase-uc2"`, `id="phase-uc3"`, `id="phase-uc4"` in the page source.

- [ ] **Step 4: Update the `refreshStatus()` JavaScript to set phase badges**

In `PORTAL_HTML`, replace the `refreshStatus` function:

**Old:**
```javascript
async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const data = await r.json();
    for (const [id, status] of Object.entries(data)) {
      const dot = document.getElementById('dot-' + id);
      if (dot) { dot.className = 'status-dot ' + status; }
    }
  } catch(e) {}
}
```

**New:**
```javascript
const PHASE_LABELS = {1: 'Phase 1', 2: 'Phase 2', 3: 'Phase 3'};
const PHASE_CLASSES = {1: 'p1', 2: 'p2', 3: 'p3'};

async function refreshStatus() {
  try {
    const r = await fetch('/api/status');
    const data = await r.json();
    for (const [id, info] of Object.entries(data)) {
      const dot = document.getElementById('dot-' + id);
      if (dot) { dot.className = 'status-dot ' + info.status; }
      const badge = document.getElementById('phase-' + id);
      if (badge) {
        const p = info.phase;
        badge.textContent = p ? PHASE_LABELS[p] || ('Phase ' + p) : (info.status === 'offline' ? 'offline' : '…');
        badge.className = 'phase-badge ' + (p ? (PHASE_CLASSES[p] || 'unknown') : 'unknown');
      }
    }
  } catch(e) {}
}
```

- [ ] **Step 5: Verify — test the status endpoint locally**

Start a quick test server (requires UC dashboards running on their ports, or use mock):
```bash
python3 -c "
import httpx, asyncio

async def check():
    async with httpx.AsyncClient(timeout=2) as c:
        r = await c.get('http://localhost:8080/api/status')
        print(r.json())

asyncio.run(check())
"
```

If no dashboards are running, each entry should show `{'status': 'offline', 'phase': None}` — that's correct.

- [ ] **Step 6: Commit**
```bash
git add kvforge_portal.py
git commit -m "feat: portal shows live phase badge (Phase 1/2/3) per use case"
```

---

## Task 4: Update deploy_and_run.sh to start vLLM for UC1/UC2/UC3

**Files:**
- Modify: `deploy_and_run.sh`

Currently the deploy script starts UC4's vLLM server implicitly via `start_uc4_dashboard.sh` but has no vLLM start for UC1/UC2/UC3. We add vLLM server launches after the pipeline workers are started. Since pipelines run in background and take 30-90+ min, the `start_vllm.sh` scripts self-exit if no checkpoint is found — so we add a separate "Step 5b" that starts vLLM servers, relying on each script's checkpoint detection.

The cleanest approach: add a Step 5b right after Step 5 that starts all 3 vLLM servers in background, each polling until a checkpoint is available (handled inside `start_vllm.sh` by the existence check).

- [ ] **Step 1: Add vLLM server launch step in deploy_and_run.sh**

Read the current `deploy_and_run.sh` and locate Step 5 (the staggered worker launch section, ending around line 125). After the UC3 staggered launch block and before `# ─── Step 6/7`, insert a new Step 5b block:

```bash
# ─── Step 5b/7 — Start vLLM inference servers for UC1, UC2, UC3 ──────────────
log "Step 5b/7 — Starting vLLM inference servers (UC1/GPU0:8091, UC2/GPU1:8092, UC3/GPU2:8093) ..."
log "  (servers will wait up to 30 min for LoRA checkpoints to appear)"

# The outer bash -c '...' keeps the SSH session open just long enough to confirm
# the background job PID before exiting cleanly — same pattern as Step 5.
# The checkpoint poll uses a plain shell directory test (no Python quoting hell).
# start_vllm.sh itself re-checks version.json and lora_checkpoints/ at launch time.

# UC1 vLLM — GPU 0, port 8091
"${SSH[@]}" "bash -c 'nohup bash -c \"
  source ~/.bashrc && source $VENV/bin/activate && cd $REMOTE_REPO
  echo [UC1 vLLM launcher] waiting for checkpoint... >> logs/vllm_uc1.log 2>&1
  for ((i=0; i<180; i++)); do
    ls -d examples/usecase1_customer_support/lora_checkpoints/v* >/dev/null 2>&1 && break
    sleep 10
  done
  CUDA_VISIBLE_DEVICES=0 bash examples/usecase1_customer_support/start_vllm.sh
\" >> $REMOTE_REPO/logs/vllm_uc1.log 2>&1 & echo UC1 vLLM launcher PID: \$!'"

sleep 2

# UC2 vLLM — GPU 1, port 8092
"${SSH[@]}" "bash -c 'nohup bash -c \"
  source ~/.bashrc && source $VENV/bin/activate && cd $REMOTE_REPO
  echo [UC2 vLLM launcher] waiting for checkpoint... >> logs/vllm_uc2.log 2>&1
  for ((i=0; i<180; i++)); do
    ls -d examples/usecase2_pubmedqa/lora_checkpoints/v* >/dev/null 2>&1 && break
    sleep 10
  done
  CUDA_VISIBLE_DEVICES=1 bash examples/usecase2_pubmedqa/start_vllm.sh
\" >> $REMOTE_REPO/logs/vllm_uc2.log 2>&1 & echo UC2 vLLM launcher PID: \$!'"

sleep 2

# UC3 vLLM — GPU 2, port 8093
"${SSH[@]}" "bash -c 'nohup bash -c \"
  source ~/.bashrc && source $VENV/bin/activate && cd $REMOTE_REPO
  echo [UC3 vLLM launcher] waiting for checkpoint... >> logs/vllm_uc3.log 2>&1
  for ((i=0; i<180; i++)); do
    ls -d examples/usecase3_squad/lora_checkpoints/v* >/dev/null 2>&1 && break
    sleep 10
  done
  CUDA_VISIBLE_DEVICES=2 bash examples/usecase3_squad/start_vllm.sh
\" >> $REMOTE_REPO/logs/vllm_uc3.log 2>&1 & echo UC3 vLLM launcher PID: \$!'"

log "Step 5b/7 — vLLM launchers started (will activate once LoRA training completes)"
```

- [ ] **Step 2: Update access URLs block at the bottom of deploy_and_run.sh**

Find the `KVForge Multi-GPU Deployment — Access URLs` echo block and add vLLM server URLs:

After the existing UC1-4 dashboard lines, add:
```bash
echo " "
echo " vLLM servers (start after LoRA training completes):"
echo "   UC1 vLLM : http://$EC2_HOST:8091  (GPU 0)"
echo "   UC2 vLLM : http://$EC2_HOST:8092  (GPU 1)"
echo "   UC3 vLLM : http://$EC2_HOST:8093  (GPU 2)"
echo "   UC4 vLLM : http://$EC2_HOST:8090  (GPU 3)"
echo " "
echo " Logs: ~/kvforge/logs/vllm_uc{1,2,3}.log"
```

- [ ] **Step 3: Verify the deploy script is valid bash**
```bash
bash -n deploy_and_run.sh && echo "Syntax OK"
```
Expected: `Syntax OK`

- [ ] **Step 4: Commit**
```bash
git add deploy_and_run.sh
git commit -m "feat: deploy_and_run.sh starts vLLM servers for UC1/UC2/UC3 after training"
```

---

## Task 5: Update run_all_pipelines.sh to start vLLM servers

**Files:**
- Modify: `run_all_pipelines.sh`

The local run script runs pipelines sequentially, so vLLM can be started immediately after each `run_pipeline` call completes (unlike the EC2 case where they run in background).

- [ ] **Step 1: Add vLLM server starts after each pipeline in run_all_pipelines.sh**

Open `run_all_pipelines.sh`. Find the three `run_pipeline` calls (they are the last lines before the empty `log ""` line that precedes the "All pipelines complete" block). Insert the new block **after** `run_pipeline 3 "usecase3_squad" "Reading Comprehension (SQuAD v2)"` and **before** the existing `log ""` / `log "════...════"` / `log "All pipelines complete. Starting dashboards..."` block. The result should look like:

```bash
run_pipeline 3 "usecase3_squad"            "Reading Comprehension (SQuAD v2)"

# ── NEW BLOCK: start vLLM servers ──────────────────────────────────────────
log ""
log "..."
# (new content below)
# ── END NEW BLOCK ──────────────────────────────────────────────────────────

log ""
log "════════════════════════════════════════"
log "All pipelines complete. Starting dashboards..."
```

Insert:

```bash
log ""
log "════════════════════════════════════════"
log "Pipelines complete. Starting vLLM servers..."
log "════════════════════════════════════════"

# Start vLLM for each UC (all 4 GPUs, non-blocking)
start_vllm() {
  local uc_num="$1"
  local dir="$2"
  local logfile="$LOGS/vllm_uc${uc_num}.log"
  nohup bash "examples/$dir/start_vllm.sh" > "$logfile" 2>&1 &
  log "vLLM UC${uc_num} started (log: $logfile)"
}

start_vllm 1 "usecase1_customer_support"
start_vllm 2 "usecase2_pubmedqa"
start_vllm 3 "usecase3_squad"
# UC4 vLLM (requires pre-trained weights at lora_checkpoints/v3/)
if [ -d "examples/usecase4_bedrock_userguide/lora_checkpoints/v3" ]; then
  start_vllm 4 "usecase4_bedrock_userguide"
else
  log "UC4 vLLM skipped — no checkpoint at examples/usecase4_bedrock_userguide/lora_checkpoints/v3"
fi

log "Waiting 30s for vLLM servers to initialize before starting dashboards..."
sleep 30
```

- [ ] **Step 2: Verify syntax**
```bash
bash -n run_all_pipelines.sh && echo "Syntax OK"
```
Expected: `Syntax OK`

- [ ] **Step 3: Commit**
```bash
git add run_all_pipelines.sh
git commit -m "feat: run_all_pipelines.sh starts vLLM servers after each pipeline completes"
```

---

## Task 6: Sync to EC2, verify, and push to GitHub

**Files:** All modified/created files above

- [ ] **Step 1: Run the deploy sync to push code to EC2**

From repo root on local machine:
```bash
EC2="ubuntu@13.221.47.200"
PEM="/Users/hemant/Downloads/RoPE/g5.x.pem"

rsync -avz --progress \
  --exclude='venv/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.faiss/' --exclude='.chroma/' --exclude='logs/' \
  --exclude='lora_checkpoints/' --exclude='.git/' \
  -e "ssh -i $PEM -o StrictHostKeyChecking=no" \
  ./ "$EC2:~/kvforge/"
```

- [ ] **Step 2: Verify new scripts are on EC2 and executable**
```bash
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 "
  ls -la ~/kvforge/examples/usecase{1,2,3}_*/start_vllm.sh
  grep -h '^PORT=\|^GPU=' ~/kvforge/examples/usecase{1,2,3}_*/start_vllm.sh
"
```
Expected:
```
-rwxr-xr-x ... usecase1_customer_support/start_vllm.sh
-rwxr-xr-x ... usecase2_pubmedqa/start_vllm.sh
-rwxr-xr-x ... usecase3_squad/start_vllm.sh
PORT=8091 / GPU=0
PORT=8092 / GPU=1
PORT=8093 / GPU=2
```

- [ ] **Step 3: Fix execute permissions on EC2 (rsync may drop +x)**
```bash
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 "
  chmod +x ~/kvforge/examples/usecase{1,2,3}_*/start_vllm.sh
  echo 'Permissions fixed'
"
```

- [ ] **Step 4: Restart UC4 portal to pick up phase changes**
```bash
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 "
  pkill -f 'kvforge_portal' 2>/dev/null || true
  sleep 1
  cd ~/kvforge
  nohup /home/ubuntu/qdrant/venv/bin/python kvforge_portal.py --port 8080 \
    >> ~/kvforge/logs/portal.log 2>&1 &
  echo 'Portal PID: '$!
"
```

- [ ] **Step 5: Verify portal shows phase data**
```bash
ssh -i /Users/hemant/Downloads/RoPE/g5.x.pem ubuntu@13.221.47.200 "
  sleep 3
  curl -s http://localhost:8080/api/status
"
```
Expected JSON (UC4 dashboard running, others offline):
```json
{
  "uc1": {"status": "offline", "phase": null},
  "uc2": {"status": "offline", "phase": null},
  "uc3": {"status": "offline", "phase": null},
  "uc4": {"status": "online", "phase": 3}
}
```

- [ ] **Step 6: Final git push to GitHub**
```bash
git log --oneline -6
git push origin smartqdrant-main
```
Expected: All 5 commits from Tasks 1-5 pushed successfully.

- [ ] **Step 7: Verify GitHub**
```bash
git log --oneline origin/smartqdrant-main | head -6
```

---

## GPU / Port Assignment Summary

| UC | GPU | vLLM Port | Dashboard Port | LoRA Name |
|----|-----|-----------|----------------|-----------|
| UC1 | 0 | 8091 | 8081 | uc1 |
| UC2 | 1 | 8092 | 8082 | uc2 |
| UC3 | 2 | 8093 | 8083 | uc3 |
| UC4 | 3 | 8090 | 8084 | uc4 |
| Portal | — | — | 8080 | — |
