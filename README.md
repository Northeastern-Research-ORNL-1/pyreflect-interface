# PyReflect Interface

A monochrome web interface for the [pyreflect](https://github.com/williamQyq/pyreflect) neutron reflectivity analysis package.

![Status](https://img.shields.io/badge/status-development-black)

## Highlights

- GitHub OAuth via NextAuth with optional MongoDB persistence for saved generations
- History explorer with search, restore, delete, and local/HF model availability
- Named runs, auto-save on generation, plus import/export of full sessions (params + results)
- Film layer editor (add/remove, collapse) with inline numeric edits beyond slider limits
- Real-time SSE logs, progress, and elapsed time during generation
- Interactive charts (NR, SLD, training loss, chi scatter) with PNG export and fullscreen expand
- Dataset/model upload for `.npy`, `.pth`, `.pt`, and `settings*.yml/.yaml`
- Model download with size lookup; optional Hugging Face dataset offload
- Local state persistence across refreshes (params, results, logs)
- Monochrome UI with JetBrains Mono and responsive layout

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
    API[FastAPI API]
    PY[pyreflect + torch]
    FS[(Local data/models)]
  end

  subgraph External
    GH[GitHub]
    DB[(MongoDB)]
    HF[(Hugging Face Dataset)]
  end

  UI <-->|REST + SSE| API
  UI <-->|Auth session| Auth
  Auth --> GH
  API --> PY
  API --> FS
  API -. optional .-> DB
  API -. optional .-> HF
```

### Generation Flow (SSE)

```mermaid
sequenceDiagram
  participant UI as Next.js UI
  participant API as FastAPI
  participant PY as pyreflect/torch
  participant DB as MongoDB (optional)
  participant HF as Hugging Face (optional)

  UI->>API: POST /api/generate/stream (params, name, X-User-ID)
  API->>PY: generate + train
  API-->>UI: SSE log/progress events
  API->>API: save model to local data/models
  API->>HF: upload model (if configured)
  API-->>UI: SSE result (nr/sld/metrics, model_id)
  API->>DB: persist history (if configured)
```

## Project Structure

```
pyreflect-interface/
├── src/
│   ├── interface/             # Next.js frontend
│   │   ├── src/app/            # App router + UI
│   │   ├── src/components/     # Panels, charts, history sidebar
│   │   ├── public/             # Static assets
│   │   └── .env.local          # Frontend secrets
│   └── backend/                # FastAPI backend
│       ├── main.py             # API server
│       ├── settings.yml        # Config (auto-generated)
│       ├── data/               # Uploaded datasets & models
│       │   ├── models/          # Saved .pth models
│       │   └── curves/          # NR/SLD curve files
│       │       └── expt/        # Experimental curves
│       └── .env                # Backend secrets
└── README.md
```

> Note: The `pyreflect` package is installed directly from GitHub rather than bundled in this repo.

## Quick Start

### Prerequisites

- Node or Bun (frontend)
- [uv](https://docs.astral.sh/uv/) (backend)
- Python 3.10-3.12 (torch requires <=3.12)

### 1. Backend Setup

```bash
cd src/backend
uv python pin 3.12
uv sync
cp .env.example .env
uv run uvicorn main:app --reload --port 8000
```

Backend runs at `http://localhost:8000`.

### 2. Frontend Setup

```bash
cd src/interface
bun install
cp .env.example .env.local
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
MAX_FILM_LAYERS=10
MAX_BATCH_SIZE=64
MAX_EPOCHS=50
MAX_CNN_LAYERS=12
MAX_DROPOUT=0.5
MAX_LATENT_DIM=32
MAX_AE_EPOCHS=100
MAX_MLP_EPOCHS=100
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

| Endpoint                    | Method | Description                             |
| --------------------------- | ------ | --------------------------------------- |
| `/api/health`               | GET    | Health check                            |
| `/api/limits`               | GET    | Current limits and production flag      |
| `/api/defaults`             | GET    | Default parameters                      |
| `/api/generate`             | POST   | Generate NR/SLD curves (non-streaming)  |
| `/api/generate/stream`      | POST   | Generate with SSE log stream            |
| `/api/status`               | GET    | Backend status and data files           |
| `/api/upload`               | POST   | Upload dataset/model files              |
| `/api/history`              | GET    | List saved generations                  |
| `/api/history`              | POST   | Save a generation manually              |
| `/api/history/{id}`         | GET    | Get full details of a save              |
| `/api/history/{id}`         | DELETE | Delete a saved generation               |
| `/api/models/{model_id}`    | GET    | Download a saved model                  |
| `/api/models/{model_id}`    | DELETE | Delete a local model file               |
| `/api/models/{model_id}/info` | GET  | Get model size and source               |

## Production Limits

Set `PRODUCTION=true` in `src/backend/.env` to enable limits.

| Parameter       | Local     | Production |
| --------------- | --------- | ---------- |
| Curves          | 100,000   | 5,000      |
| Film Layers     | 20        | 10         |
| Batch Size      | 512       | 64         |
| Epochs          | 1,000     | 50         |
| CNN Layers      | 20        | 12         |
| Dropout         | 0.9       | 0.5        |
| Latent Dim      | 128       | 32         |
| AE Epochs       | 500       | 100        |
| MLP Epochs      | 500       | 100        |

## Model Storage Notes

- Local storage keeps up to 2 models; runs will fail if the limit is reached.
- Set `HF_TOKEN` and `HF_REPO_ID` to offload models to Hugging Face and auto-clean local files.

## Troubleshooting

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9
```
