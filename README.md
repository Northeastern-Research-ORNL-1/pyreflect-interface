# PyReflect Interface

A minimal, monochrome web interface for the [pyreflect](https://github.com/williamQyq/pyreflect) neutron reflectivity analysis package.

![Interface Preview](https://img.shields.io/badge/status-development-black)
![Version](https://img.shields.io/badge/version-v0.0.3-black)

## Version

- **v0.0.3** — Added history explorer, automatic named saving, and deletion functionality.
- **v0.0.2** — Added GitHub OAuth, MongoDB persistence, SSE heartbeats, and info tooltips.
- **v0.0.1** — Initial GUI release with streaming backend, charts, and uploads.
  - **NOTE** — Curve set to 1000 default globally

## Features

- **GitHub Authentication**: Sign in with GitHub to save and track your generations
- **History Explorer**: Browse, search, and restore past generations
- **Automatic Persistence**: Generations are automatically saved with custom names to MongoDB
- **Info Tooltips**: Hover over (ⓘ) icons to learn what each parameter does
- **SSE Heartbeats**: Prevents Cloudflare proxy timeouts during long training runs
- **Adjustable Parameters**: Film layers (SLD, thickness, roughness), generator settings, training configuration
- **Ground Truth vs Predicted**: NR and SLD charts show both ground truth (solid) and model predictions (dashed)
- **Graph Visualization**: Downloadable & interactive NR curves, SLD profiles, training loss, Chi parameter scatter plots
- **Monochrome Design**: Clean black/white aesthetic with JetBrains Mono font
- **Real-time Updates**: Instant parameter feedback with generate-on-demand
- **Editable Values**: Click any numeric value to type custom inputs beyond slider limits
- **Live Streaming Logs**: Real-time training progress streamed from backend via SSE
- **Timing + Warnings**: Generation/training/inference timings and backend warnings streamed to console
- **Data Upload**: Drag-and-drop upload for `.npy` datasets and `.pth` model weights
- **State Persistence**: Parameters and results persist across browser refreshes
- **Reset + Collapse**: One-click reset to example defaults and per-layer collapse/expand controls

## Project Structure

```
pyreflect-interface/
├── src/
│   ├── interface/          # Next.js frontend
│   │   ├── src/app/
│   │   │   ├── api/auth/   # NextAuth GitHub OAuth
│   │   │   └── page.tsx    # Main app
│   │   └── .env.local      # Frontend secrets
│   └── backend/            # FastAPI backend
│       ├── main.py         # API server
│       ├── settings.yml    # Config (auto-generated)
│       ├── .env            # Backend secrets
│       └── data/           # Uploaded datasets & models
│           └── curves/     # NR/SLD curve files
└── README.md
```

> **Note**: The `pyreflect` package is installed directly from [GitHub](https://github.com/williamQyq/pyreflect) rather than bundled in this repo.

## Quick Start

### Prerequisites

- [Bun](https://bun.sh) or [npm](https://nodejs.org) (frontend)
- [uv](https://docs.astral.sh/uv/) (backend)
- Python 3.10-3.12 (torch requires ≤3.12)

### 1. Backend Setup

```bash
cd src/backend
uv python pin 3.12      # Pin to Python 3.12 (required for torch)
uv sync                 # Install dependencies
cp .env.example .env    # Configure environment
uv run uvicorn main:app --reload --port 8000
```

Backend runs at **http://localhost:8000**

### 2. Frontend Setup

```bash
cd src/interface
npm install             # or: bun install
cp .env.example .env.local
npm run dev             # or: bun dev
```

Frontend runs at **http://localhost:3000**

### 3. Environment Variables

**Backend (`src/backend/.env`):**

```env
PRODUCTION=false
CORS_ORIGINS=http://localhost:3000,https://pyreflect.shlawg.com
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net/?appName=shlawg
```

**Frontend (`src/interface/.env.local`):**

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-secret-here   # Generate with: openssl rand -base64 32
GITHUB_CLIENT_ID=your-github-oauth-app-id
GITHUB_CLIENT_SECRET=your-github-oauth-app-secret
```

## GitHub OAuth Setup

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Create a new OAuth App:
   - **Homepage URL**: `http://localhost:3000` (or production URL)
   - **Authorization callback URL**: `http://localhost:3000/api/auth/callback/github`
3. Copy Client ID and Client Secret to `.env.local`

## MongoDB Setup

1. Create a [MongoDB Atlas](https://www.mongodb.com/atlas) cluster
2. Get your connection string (with username/password)
3. Add to `MONGODB_URI` in backend `.env`
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
    "nr": { "q": [...], "reflectivity": [...] },
    "sld": { "z": [...], "sld": [...] },
    "training": { "epochs": [...], "loss": [...] },
    "chi": [...],
    "metrics": { ... }
  }
}
```

## API Endpoints

| Endpoint               | Method | Description                             |
| ---------------------- | ------ | --------------------------------------- |
| `/api/health`          | GET    | Health check                            |
| `/api/generate`        | POST   | Generate NR/SLD curves                  |
| `/api/generate/stream` | POST   | Generate with real-time SSE log stream  |
| `/api/defaults`        | GET    | Get default parameters                  |
| `/api/status`          | GET    | Backend status and available data files |
| `/api/limits`          | GET    | Get current parameter limits            |
| `/api/upload`          | POST   | Upload dataset/model files              |
| `/api/history`         | GET    | Get list of saved generations           |
| `/api/history/{id}`    | GET    | Get full details of a save              |
| `/api/history/{id}`    | DELETE | Delete a saved generation               |

## Production Deployment

### Backend Limits

| Parameter     | Local   | Production |
| ------------- | ------- | ---------- |
| Curves        | 100,000 | 5,000      |
| Epochs        | 1,000   | 50         |
| Batch Size    | 512     | 64         |
| CNN Layers    | 20      | 12         |
| Dropout       | 0.9     | 0.5        |
| Latent Dim    | 128     | 32         |
| AE/MLP Epochs | 500     | 100        |

Set `PRODUCTION=true` in backend `.env` to enable limits.

### Vercel Deployment (Frontend)

```bash
cd src/interface
vercel
```

Set environment variables in Vercel dashboard:

- `NEXT_PUBLIC_API_URL` - Backend URL
- `NEXTAUTH_URL` - Frontend URL
- `NEXTAUTH_SECRET` - Random secret
- `GITHUB_CLIENT_ID` - OAuth app ID
- `GITHUB_CLIENT_SECRET` - OAuth app secret

### Troubleshooting

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Check Python version (must be 3.10-3.12)
python --version

# Force Python 3.12 for backend
uv python pin 3.12
uv sync
```

## Technology Stack

- **Frontend**: Next.js 16, React 19, TypeScript, Recharts, NextAuth.js
- **Backend**: FastAPI, Pydantic, NumPy, PyMongo
- **ML Package**: pyreflect (PyTorch, refl1d, refnx)
- **Database**: MongoDB Atlas
- **Auth**: GitHub OAuth via NextAuth

## Credits

- [pyreflect](https://github.com/williamQyq/pyreflect) - NR-SCFT-ML package by Yuqing Qiao
- Based on research by Brian Qu, Dr. Rajeev Kumar, Prof. Miguel Fuentes-Cabrera
