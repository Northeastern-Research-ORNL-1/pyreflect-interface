# PyReflect Interface

A monochrome web interface for the [pyreflect](https://github.com/williamQyq/pyreflect) neutron reflectivity analysis package.

![Status](https://img.shields.io/badge/status-development-black)
![Version](https://img.shields.io/badge/version-v0.1.0-black)

## Highlights

- GitHub OAuth via NextAuth with MongoDB history persistence
- Tool hints for most fields on hover
- Two data modes: `Synthetic (film layers)` and `Real data (.npy via settings.yml)`
- Real-data pipelines: `NR → SLD` (train/infer), `SLD → Chi`, and `NR → SLD → Chi` (chains predicted SLD into chi)
- Role-based uploads that update `src/backend/settings.yml`, plus a UI mapping view to show what each role points to
- Live SSE logs + per-epoch training progress (header + in-history “in progress” card)
- Export JSON (params + result + embedded chart PNGs) and download a ZIP bundle (JSON + PNGs + model)
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
  FS --> FS_MODELS
  FS --> FS_CURVES
  FS_CURVES --> FS_EXPT
  API -. "optional" .-> DB
  API -. "optional" .-> HF
```

### Synthetic Workflow (Film Layers)

```mermaid
flowchart TD
  UI[UI: film layers + params] --> API[POST /api/generate/stream]
  API --> GEN[pyreflect: ReflectivityDataGenerator]
  GEN --> SYN["Synthetic NR/SLD arrays (in-memory)"]
  SYN --> TRAIN[Train CNN NR → SLD]
  TRAIN --> SAVE[Save model_id.pth]
  SAVE --> FS_MODELS[data/models/]
  SAVE -. "optional" .-> HF[(Hugging Face Dataset)]
  TRAIN --> RESULT[result JSON + model_id]
  RESULT --> UI
  RESULT -. "optional" .-> DB["(MongoDB history)"]
  API -- "SSE log/progress/result" --> UI
```

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
  H[Header buttons] -->|History| HS[Open History sidebar]
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

- `History` → opens the history sidebar (`src/interface/src/components/ExploreSidebar.tsx`).
- `JSON` VIEW menu → `Import` / `Export`.
- `Download` → opens the download confirmation modal (`src/interface/src/components/DownloadBundleModal.tsx`).
- `Profile` → `Sign in` / `Sign out` → GitHub OAuth via NextAuth.

### Left Panel (Parameters)

```mermaid
flowchart TD
  P[Parameter panel buttons] -->|RESET| RST[Reset params + clear current run]
  P -->|GENERATE| GEN[Start generation]
  GEN -->|needs name| POP[Name popup]
  POP -->|START| RUN["POST /api/generate/stream (SSE)"]
  POP -->|CANCEL| STOP[Close popup]
  RUN --> LOGS[Stream logs + progress]
  RUN --> RES[Render charts + metrics]
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
  HS -->|Download icon| DLREQ[Load + open download confirm]
  DLREQ --> BUILD[Build ZIP in browser]
  HS -->|Delete| CONF[Delete confirm popup]
  CONF -->|DELETE| DEL["DELETE /api/history/{id}"]
  DEL --> REFRESH[Refresh list]
```

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
| `/api/history/{id}`           | DELETE | Delete a saved generation and its model |
| `/api/models/{model_id}`      | GET    | Download a saved model                  |
| `/api/models/{model_id}`      | DELETE | Delete a local model file               |
| `/api/models/{model_id}/info` | GET    | Get model size and source               |

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

- Synthetic training keeps up to 2 local models; runs will fail if the limit is reached.
- Set `HF_TOKEN` and `HF_REPO_ID` to offload models to Hugging Face and auto-clean local files.
- Deleting a history item also deletes its model file locally and from Hugging Face (if configured).

## Troubleshooting

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```
