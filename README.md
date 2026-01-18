# PyReflect Interface

A monochrome web interface for the [pyreflect](https://github.com/williamQyq/pyreflect) neutron reflectivity analysis package.

![Status](https://img.shields.io/badge/status-development-black)
![Version](https://img.shields.io/badge/version-v0.1.1-black)

## Highlights

- GitHub OAuth via NextAuth with MongoDB history persistence
- Tool hints for most fields on hover
- Two data modes: `Synthetic (film layers)` and `Real data (.npy via settings.yml)`
- Real-data pipelines: `NR → SLD` (train/infer), `SLD → Chi`, and `NR → SLD → Chi` (chains predicted SLD into chi)
- Role-based uploads that update `src/backend/settings.yml`, plus a UI mapping view to show what each role points to
- Streaming SSE endpoint for synchronous logs/progress (`/api/generate/stream`)
- Export JSON (params + result + embedded chart PNGs) and download a ZIP bundle (JSON + PNGs + model)
- Redis-backed queue for non-blocking, concurrent synthetic training submissions. The "Explore" panel shows real-time jobs (elapsed time + last log line), supports rename (double-click) + cancel (queued) + retry/delete (failed), and auto-claims guest-started jobs when you log in so results save to history.
- Model Object Storage via [Hugging Face dataset](https://huggingface.co/datasets/Northeastern-Research-ORNL-1/models/tree/main) + size lookup and download redirect

## Architecture

### System Overview

```mermaid
flowchart LR
  subgraph Browser
    UI[Next.js UI]
  end

  subgraph NextApp[Next.js App]
    Auth[NextAuth GitHub OAuth]
  end

  subgraph Backend
    API["FastAPI (service)"]
    PY[pyreflect + torch]
    CFG[settings.yml]
    FS[(data/)]
    FS_MODELS[data/models/*.pth]
    FS_CURVES[data/curves/*.npy]
    FS_EXPT[data/curves/expt/*.npy]
  end

  subgraph Queue["Job Queue (optional)"]
    REDIS[(Redis)]
    RQ[RQ Worker]
  end

  subgraph External
    GH[GitHub]
    DB[(MongoDB)]
    HF[(Hugging Face Dataset)]
  end

  UI <-->|"REST + SSE"| API
  UI <-->|"Auth session"| Auth
  Auth --> GH
  API --> PY
  API --> CFG
  API --> FS
  API -.-> REDIS
  REDIS --> RQ
  RQ --> PY
  FS --> FS_MODELS
  FS --> FS_CURVES
  FS_CURVES --> FS_EXPT
  API -. "optional" .-> DB
  API -. "optional" .-> HF
```

### Synthetic Workflow (Film Layers)

```mermaid
flowchart TD
  UI[UI: film layers + params] --> API[POST /api/jobs/submit]
  API --> REDIS[(Redis)]
  REDIS --> RQ[RQ Worker]
  RQ --> GEN[pyreflect: ReflectivityDataGenerator]
  GEN --> SYN["Synthetic NR/SLD arrays (in-memory)"]
  SYN --> TRAIN[Train CNN NR → SLD]
  TRAIN --> SAVE[Save model_id.pth]
  SAVE --> FS_MODELS[data/models/]
  SAVE -. "optional" .-> HF[(Hugging Face Dataset)]
  TRAIN --> RESULT[result JSON + model_id]
  RESULT -. "optional" .-> DB["(MongoDB history)"]
```

> Synchronous runs are still available via `POST /api/generate/stream` (SSE), but the current UI submits via the job queue.

### Real-Data Workflow

```mermaid
flowchart TD
  UI[UI: real-data mode] -->|Upload roles| UP[POST /api/upload]
  UP -->|Write files| FS[(data/...)]
  UP -->|Update| CFG[settings.yml]

  UI --> API[POST /api/generate/stream]
  API --> CFG
  CFG --> ROUTE{pipeline}

  ROUTE -->|"NR → SLD"| NRSLD{mode}
  NRSLD -->|"Train"| NRTRAIN["train CNN on nr_train + sld_train<br/>auto-generate model + stats if missing"]
  NRTRAIN --> SLDOUT[predicted SLD]
  NRSLD -->|"Infer"| NRINFER["predict SLD from experimental_nr<br/>using model + normalization stats"]
  NRINFER --> SLDOUT

  ROUTE -->|"SLD → Chi"| SLDCHI["train AE+MLP on model_sld_file + model_chi_params_file<br/>predict chi for model_experimental_sld_profile"]
  SLDCHI --> CHIOUT[chi params]

  ROUTE -->|"NR → SLD → Chi"| CHAIN["use predicted SLD as chi input<br/>train AE+MLP on model_sld_file + model_chi_params_file"]
  SLDOUT --> CHAIN
  CHAIN --> CHIOUT

  SLDOUT --> RESULT[result JSON]
  CHIOUT --> RESULT
  RESULT --> UI
```

> Real-data mode uses the paths in `src/backend/settings.yml`. Film layers and generator settings are ignored.

### Required Uploads By Pipeline (Real Data)

- `NR → SLD (train)`: `nr_train`, `sld_train` (+ `nr_sld_model`, `normalization_stats` if auto-generate is disabled)
- `NR → SLD (infer)`: `experimental_nr`, `nr_sld_model`, `normalization_stats`
- `SLD → Chi`: `sld_chi_experimental_profile`, `sld_chi_model_sld_file`, `sld_chi_model_chi_params_file`
- `NR → SLD → Chi (train)`: `nr_train`, `sld_train`, `sld_chi_model_sld_file`, `sld_chi_model_chi_params_file`  
  (chains predicted SLD into chi, so `sld_chi_experimental_profile` is not required)
- `NR → SLD → Chi (infer)`: `experimental_nr`, `nr_sld_model`, `normalization_stats`, `sld_chi_model_sld_file`, `sld_chi_model_chi_params_file`

### Upload Flow

```mermaid
flowchart LR
  UI[UI: choose role + file] --> API[POST /api/upload]
  API -->|"validate shapes (.npy)"| VALID[server-side validation]
  API -->|"write TEMP file to disk"| FS[(data/...)]
  API -->|"update role path"| CFG[settings.yml]
```

Notes:

- `/api/upload` expects an explicit `roles[]` entry for each uploaded file. The backend does not guess roles from filenames, because filename-based inference is ambiguous and can write the wrong `settings.yml` mapping.
- Uploading `settings.yml` directly is supported (filename must start with `settings`), but normal `.npy/.pth` uploads should always include a role.

### Upload Roles (Real Data)

When uploading datasets, assign a role so `settings.yml` is updated correctly:

- `nr_train` → `nr_predict_sld.file.nr_train`
- `sld_train` → `nr_predict_sld.file.sld_train`
- `experimental_nr` → `nr_predict_sld.file.experimental_nr_file`
- `normalization_stats` → `nr_predict_sld.models.normalization_stats`
- `nr_sld_model` → `nr_predict_sld.models.model`
- `sld_chi_experimental_profile` → `sld_predict_chi.file.model_experimental_sld_profile`
- `sld_chi_model_sld_file` → `sld_predict_chi.file.model_sld_file`
- `sld_chi_model_chi_params_file` → `sld_predict_chi.file.model_chi_params_file`

### Expected `.npy` Shapes (Real Data)

Uploads are validated on the backend. Current expectations:

- `nr_train`, `sld_train`, `experimental_nr`: `shape (N, 2, L)` where axis 1 is `[x, y]`
- `sld_chi_experimental_profile`, `sld_chi_model_sld_file`: `shape (2, L)` or `shape (N, 2, L)`
- `sld_chi_model_chi_params_file`: `shape (N, num_params)`
- `normalization_stats`: a pickled dict with keys `x` and `y` (min/max), used for normalization

## Exports & Downloads

- **Export JSON**: includes `params` + `result` and embeds normal + expanded chart PNGs in `result.export_pngs` (base64) when available.
- **Download bundle**: builds a `.zip` in the browser with your selected items (`output.json`, PNGs, and the `.pth` model). If the model isn’t local, `/api/models/{model_id}` redirects to Hugging Face (when configured).

> The app currently exports the plotted curves/metrics for a run, not the full synthetic training dataset arrays.

## UI Button Flows

This section documents what each major UI button does and which code path it triggers.

### Header (Top Bar)

```mermaid
flowchart LR
  H[Header buttons] -->|Explore| HS[Open Explore sidebar]
  HS -->|fetch| APIH[GET /api/history]
  H -->|JSON ▾ → Import| IMP[Import JSON]
  IMP --> FP[Choose .json file]
  FP --> PARSE[Parse + load params/result]
  PARSE --> NAME[Optional: name modal]
  NAME --> APPLY[Apply to UI state]
  H -->|JSON ▾ → Export| EXP[Export JSON]
  EXP --> CAP["captureChartPngs (card view)"]
  CAP --> DLJSON[Download .json]
  H -->|Download| DB[Open bundle confirm modal]
  DB -->|DOWNLOAD| BUILD[Build ZIP in browser]
  BUILD --> CAP2["captureChartPngs (includes fullscreen-expanded when selected)"]
  CAP2 --> MODEL["Optional: fetch /api/models/{model_id}"]
  MODEL --> DLZIP[Download .zip]
```

**Buttons**

- `Explore` → opens the Explore sidebar (Queue + History) (`src/interface/src/components/ExploreSidebar.tsx`).
- `JSON` VIEW menu → `Import` / `Export`.
- `Download` → opens the download confirmation modal (`src/interface/src/components/DownloadBundleModal.tsx`).
- `Profile` → `Sign in` / `Sign out` → GitHub OAuth via NextAuth.

### Left Panel (Parameters)

```mermaid
flowchart TD
  P[Parameter panel buttons] -->|RESET| RST[Reset params + clear current run]
  P -->|GENERATE| GEN[Start generation]
  GEN -->|needs name| POP[Name popup]
  POP -->|START| RUN["POST /api/jobs/submit (queue, non-blocking)"]
  POP -->|CANCEL| STOP[Close popup]
  RUN --> QLOG[Log job id + show in Explore]
  P -->|+ ADD| ADDL[Add film layer]
  P -->|×| RML[Remove film layer]
  P -->|EXPAND/COLLAPSE| LAY[Expand/collapse all layers]
  P -->|▶/▼ per-layer| LAY1[Expand/collapse single layer]
  P -->|◀/▶| SIDE[Collapse/expand sidebar]
```

### Charts (Right Panel)

```mermaid
flowchart LR
  C[Chart card] -->|PNG| PNG[Capture card to PNG]
  PNG --> DL[Download *.png]
  C -->|Expand| X[Fullscreen expand]
  X -->|Close / backdrop| XC[Return to grid]
```

### Console Logs

```mermaid
flowchart LR
  CL[Console Logs header] -->|▶/▼| TOG[Toggle log panel]
  TOG -->|expanded| SCROLL[Auto-scroll to newest]
```

### History Sidebar

```mermaid
flowchart TD
  HS[History item] -->|click row| LOAD["GET /api/history/{id}"]
  LOAD --> APPLY[Apply to UI state]
  HS -->|double-click rename| REN["PATCH /api/history/{id}"]
  HS -->|Download icon| DLREQ[Load + open download confirm]
  DLREQ --> BUILD[Build ZIP in browser]
  HS -->|Delete| CONF[Delete confirm popup]
  CONF -->|DELETE| DEL["DELETE /api/history/{id}"]
  DEL --> REFRESH[Refresh list]
```

Notes:

- History + Queue names can be renamed inline (double-click → edit → blur/Enter saves).
- Explore includes a `Clear job cache` button to remove your finished/failed job records from Redis (does not delete history).

### Download Confirmation Modal

```mermaid
flowchart TD
  M[Download modal] -->|CANCEL| CLOSE[Close modal]
  M -->|DOWNLOAD| START[Close modal + show downloading overlay]
  START --> CAP[Capture selected PNGs]
  CAP --> ZIP[Create ZIP]
  ZIP --> SAVE[Trigger browser download]
```

## Project Structure

```
pyreflect-interface/
├── src/
│   ├── interface/             # Next.js frontend
│   │   ├── src/app/            # App router + UI
│   │   │   └── home/           # HomePage + hooks (downloads, etc)
│   │   ├── src/components/     # Panels, charts, history sidebar
│   │   ├── src/lib/            # Small shared helpers (PNG capture, base64, bytes)
│   │   ├── public/             # Static assets
│   │   └── .env.local          # Frontend secrets
│   └── backend/                # FastAPI backend
│       ├── main.py             # Uvicorn entrypoint
│       ├── service/            # App factory + routers + services
│       ├── settings.yml        # Config (auto-generated)
│       ├── data/               # Uploaded datasets & models
│       │   ├── models/          # Saved .pth models
│       │   └── curves/          # NR/SLD curve files
│       │       └── expt/        # Experimental curves
│       └── .env                # Backend secrets
└── README.md
```

> Note: The `pyreflect` package is installed directly from GitHub rather than bundled in this repo.

## Quick Start Self Hosted

### Prerequisites

- Bun (frontend)
- [uv](https://docs.astral.sh/uv/) (backend)
- Python 3.10-3.12 (torch requires <=3.12)

### 1. Backend Setup

```bash
cd src/backend
uv python pin 3.12
uv sync
cp .env.example .env
nano/nvim .env
uv run uvicorn main:app --reload --port 8000
```

Backend runs at `http://localhost:8000`.

#### Job Queue (Optional)

For non-blocking job submission (allows multiple concurrent generations):

```bash
# Install Redis (one-time)
brew install redis  # macOS
sudo apt install redis-server  # Ubuntu

# Start Redis before running uvicorn
redis-server --daemonize yes
```

When Redis is running, the backend will automatically:

1. Connect to Redis on startup
2. Spawn an RQ worker subprocess
3. Accept job submissions via `/api/jobs/submit`

Jobs are queued and processed in the background. Results appear in history when complete.

> Note: the current UI submits work via the job queue (`/api/jobs/submit`). If Redis/RQ isn’t available you’ll see a “Queue not available” message; you can still call `/api/generate/stream` manually (curl/Postman) for a synchronous run.

### 2. Frontend Setup

```bash
cd src/interface
bun install
cp .env.example .env.local
nano/nvim .env.local
bun run dev
```

Frontend runs at `http://localhost:3000`.

## Environment Variables

### Backend (`src/backend/.env`)

```env
PRODUCTION=false
CORS_ORIGINS=http://localhost:3000,https://pyreflect.shlawg.com

# MongoDB (optional)
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/?appName=shlawg

# Hugging Face (optional model offload)
HF_TOKEN=hf_...
HF_REPO_ID=your-username/pyreflect-models

# Production limits (only used when PRODUCTION=true)
MAX_CURVES=5000
...

# Redis Queue (optional, for job queuing)
REDIS_URL=redis://localhost:6379
# RQ timeout for training jobs (e.g. 30m, 2h)
RQ_JOB_TIMEOUT=2h

# Local model storage (MODELS_DIR/*.pth)
# If the folder is full, runs will wait instead of erroring.
MAX_LOCAL_MODELS=2
# Seconds to wait for a slot (0 = wait forever)
LOCAL_MODEL_WAIT_TIMEOUT_S=900
# Poll interval while waiting
LOCAL_MODEL_WAIT_POLL_S=2.0
```

### Frontend (`src/interface/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-here
GITHUB_CLIENT_ID=your-github-oauth-app-id
GITHUB_CLIENT_SECRET=your-github-oauth-app-secret
```

## GitHub OAuth Setup

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Create a new OAuth App:
   - Homepage URL: `http://localhost:3000` (or production URL)
   - Authorization callback URL: `http://localhost:3000/api/auth/callback/github`
3. Copy Client ID and Client Secret to `.env.local`

## MongoDB Setup

1. Create a [MongoDB Atlas](https://www.mongodb.com/atlas) cluster
2. Get your connection string (with username/password)
3. Add it to `MONGODB_URI` in `src/backend/.env`
4. The `generations` collection is created automatically on first save

### MongoDB Document Structure

Each saved generation contains:

```json
{
  "_id": "ObjectId(...)",
  "user_id": "12345678",
  "name": "My Experiment 1",
  "created_at": "2026-01-16T04:23:14Z",
  "params": {
    "layers": [...],
    "generator": { "numCurves": 1000, ... },
    "training": { "epochs": 10, "batchSize": 32, ... }
  },
  "result": {
    "nr": { "q": [...], "groundTruth": [...], "computed": [...] },
    "sld": { "z": [...], "groundTruth": [...], "predicted": [...] },
    "training": { "epochs": [...], "trainingLoss": [...], "validationLoss": [...] },
    "chi": [...],
    "metrics": { "mse": 0.0, "r2": 0.0, "mae": 0.0 },
    "model_id": "uuid"
  }
}
```

## API Endpoints

| Endpoint                      | Method | Description                             |
| ----------------------------- | ------ | --------------------------------------- |
| `/api/health`                 | GET    | Health check                            |
| `/api/limits`                 | GET    | Current limits and production flag      |
| `/api/defaults`               | GET    | Default parameters                      |
| `/api/generate`               | POST   | Generate NR/SLD curves (non-streaming)  |
| `/api/generate/stream`        | POST   | Generate with SSE log stream            |
| `/api/status`                 | GET    | Backend status and data files           |
| `/api/upload`                 | POST   | Upload files (+ optional roles)         |
| `/api/history`                | GET    | List saved generations                  |
| `/api/history`                | POST   | Save a generation manually              |
| `/api/history/{id}`           | GET    | Get full details of a save              |
| `/api/history/{id}`           | PATCH  | Rename a saved generation               |
| `/api/history/{id}`           | DELETE | Delete a saved generation and its model |
| `/api/models/{model_id}`      | GET    | Download a saved model                  |
| `/api/models/{model_id}`      | DELETE | Delete a local model file               |
| `/api/models/{model_id}/info` | GET    | Get model size and source               |
| `/api/jobs/submit`            | POST   | Submit job to queue (non-blocking)      |
| `/api/jobs/{job_id}`          | GET    | Get job status, progress, and result    |
| `/api/jobs/{job_id}`          | DELETE | Cancel a queued job                     |
| `/api/jobs/{job_id}/name`     | PATCH  | Rename a queued job                     |
| `/api/jobs/{job_id}/retry`    | POST   | Retry a failed/finished job             |
| `/api/jobs/{job_id}/delete`   | DELETE | Delete a job record (non-running only)  |
| `/api/jobs/{job_id}/claim`    | POST   | Attach a job to a user (login mid-run)  |
| `/api/jobs/purge`             | DELETE | Delete non-running jobs for a user      |
| `/api/queue`                  | GET    | Queue status and worker info            |

## Production Limits

Set `PRODUCTION=true` in `src/backend/.env` to enable limits.

| Parameter   | Local   | Production |
| ----------- | ------- | ---------- |
| Curves      | 100,000 | 5,000      |
| Film Layers | 20      | 10         |
| Batch Size  | 512     | 64         |
| Epochs      | 1,000   | 50         |
| CNN Layers  | 20      | 12         |
| Dropout     | 0.9     | 0.5        |
| Latent Dim  | 128     | 32         |
| AE Epochs   | 500     | 100        |
| MLP Epochs  | 500     | 100        |

## Model Storage Notes

- Synthetic training keeps up to `MAX_LOCAL_MODELS` local `*.pth` files under `src/backend/data/models/`. When the limit is reached, new runs will wait (up to `LOCAL_MODEL_WAIT_TIMEOUT_S`) for a slot to free up.
  - Any `*.pth` file counts (including test/dummy files).
- Set `HF_TOKEN` and `HF_REPO_ID` to offload models to Hugging Face and auto-clean local files.
- Deleting a history item also deletes its model file locally and from Hugging Face (if configured).

## Troubleshooting

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```

Common issues:

- **Job looks stuck on “Saving model…”**: the local models folder is full (`src/backend/data/models/*.pth`) or `torch.save` is slow. Check the last log line in Explore; delete/move a `*.pth` file or bump `MAX_LOCAL_MODELS`.
- **Jobs marked failed after ~30m**: increase `RQ_JOB_TIMEOUT` (the default is `2h`).
- **Guest → login mid-job**: Explore will attempt to “claim” the running job so the worker can save to Mongo history when it finishes (`/api/jobs/{job_id}/claim`).

## Job Queue Architecture

When Redis is available, the system supports queued job execution:

```mermaid
sequenceDiagram
    participant UI as Frontend
    participant API as FastAPI
    participant Redis as Redis Queue
    participant RQ as RQ Worker
    participant DB as MongoDB

    UI->>API: POST /api/jobs/submit
    API->>Redis: Enqueue job
    API-->>UI: {job_id, status: "queued"}
    Note over UI: UI can submit more jobs

    RQ->>Redis: Fetch job
    RQ->>RQ: Run training
    RQ->>DB: Save result

    UI->>API: GET /api/jobs/{job_id}
    API->>Redis: Get job status
    API-->>UI: {status, progress, result}
```

### Job API Examples

**Submit a job:**

```bash
curl -X POST http://localhost:8000/api/jobs/submit \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user123" \
  -d '{"name": "My Job", "layers": [], "generator": {"numCurves": 1000}, "training": {"epochs": 50}}'
```

**Check job status:**

```bash
curl http://localhost:8000/api/jobs/{job_id}
```

Response includes `status` (queued/started/finished/failed), `meta.progress`, and `result` when complete.

**Rename a job:**

```bash
curl -X PATCH http://localhost:8000/api/jobs/{job_id}/name \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user123" \
  -d '{"name": "New name"}'
```

**Rename a history item:**

```bash
curl -X PATCH http://localhost:8000/api/history/{id} \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user123" \
  -d '{"name": "New name"}'
```

**Clear job cache (finished/failed jobs):**

```bash
curl -X DELETE http://localhost:8000/api/jobs/purge \
  -H "X-User-Id: user123"
```

### Additional Workers

To process multiple jobs in parallel:

```bash
uv run rq worker training --path .                # Single worker
uv run rq worker-pool training -n 2 --path .      # Worker pool
```

### Google Colab GPU Worker

Use `colab_gpu_worker.ipynb` to process training jobs on Colab's free GPU:

1. Upload the notebook to [Google Colab](https://colab.research.google.com)
2. Update `REDIS_URL` with your VPS credentials: `redis://:PASSWORD@YOUR_VPS_IP:6379`
3. Enable GPU runtime: `Runtime > Change runtime type > T4 GPU`
4. Run all cells — the worker will connect to your queue and process jobs

**VPS Redis setup:**

```bash
# Edit redis.conf
sudo nano /etc/redis/redis.conf

# Add/uncomment these lines:
bind 0.0.0.0
requirepass YOUR_STRONG_PASSWORD

# Restart Redis
sudo systemctl restart redis

# Open firewall (if needed)
sudo ufw allow 6379
```

Update your backend `.env` to use the password:

```env
REDIS_URL=redis://:YOUR_PASSWORD@localhost:6379
```

> Note: Colab notebooks are private by default, so your Redis password is safe unless you share the notebook.

### Monitoring

```bash
uv run rq info              # Queue status
uv run rq info --by-queue   # Jobs by queue
```

Optional dashboard:

```bash
uv add rq-dashboard && uv run rq-dashboard   # http://localhost:9181
```

**Fallback Behavior (UI)**: The current UI does not auto-fallback; it will log that the queue is unavailable. Use `/api/generate/stream` directly if you want synchronous execution without Redis.
