# Architecture

**Analysis Date:** 2026-01-20

## Pattern Overview

**Overall:** Monorepo with Next.js frontend and FastAPI backend (client-server architecture)

**Key Characteristics:**
- Next.js App Router UI with client components (`src/interface/src/app`, `src/interface/src/components`)
- FastAPI backend with modular routers (`src/backend/main.py`, `src/backend/service/routers/`)
- Server-Sent Events for long-running generation (`/api/generate/stream`)
- Redis + RQ job queue for background training with Modal GPU workers
- Checkpointing service for pause/resume across worker crashes
- Optional persistence to MongoDB and Hugging Face

## Layers

**UI Layer:**
- Purpose: User interface, auth, and local state
- Contains: App router pages, UI components, CSS Modules
- Depends on: API endpoints and NextAuth session
- Used by: Browser runtime
- Locations: `src/interface/src/app`, `src/interface/src/components`

**Frontend Types Layer:**
- Purpose: Shared typing for API data
- Contains: Interfaces and shared types
- Depends on: None (type-only)
- Used by: UI components and page logic
- Location: `src/interface/src/types/index.ts`

**Auth Layer:**
- Purpose: GitHub OAuth session for the UI
- Contains: NextAuth handler and callbacks
- Depends on: NextAuth GitHub provider
- Used by: UI for session state
- Location: `src/interface/src/app/api/auth/[...nextauth]/route.ts`

**API Layer:**
- Purpose: HTTP endpoints for generation, uploads, jobs, and history
- Contains: FastAPI routes and response models
- Depends on: Job queue, checkpointing service, storage adapters
- Used by: UI via REST and SSE
- Locations: `src/backend/main.py`, `src/backend/service/routers/`

**Job Queue Layer:**
- Purpose: Background job execution with Redis + RQ
- Contains: Job submission, status tracking, worker coordination
- Depends on: Redis, Modal GPU workers
- Used by: API layer for non-blocking training
- Locations: `src/backend/service/routers/jobs.py`, `src/backend/service/jobs/`

**Checkpointing Layer:**
- Purpose: Training state persistence for crash recovery and pause/resume
- Contains: Checkpoint save/load/delete, HuggingFace Hub integration
- Depends on: HuggingFace Hub API, PyTorch
- Used by: Training job function
- Location: `src/backend/service/services/checkpointing.py`

**Processing Layer:**
- Purpose: Curve generation, training, inference pipelines
- Contains: pyreflect pipelines, numpy processing, streaming output
- Depends on: pyreflect, numpy, torch
- Used by: API endpoints and job workers
- Locations: `src/backend/main.py`, `src/backend/service/jobs/__init__.py`

**Persistence Layer:**
- Purpose: Optional history storage and model persistence
- Contains: MongoDB access, local filesystem storage, Hugging Face offload
- Depends on: pymongo, filesystem, huggingface_hub
- Used by: API endpoints
- Locations: `src/backend/main.py`, `src/backend/data/`

## Data Flow

**Generation (SSE):**
1. UI submits request from `src/interface/src/app/page.tsx`
2. Backend handles `POST /api/generate/stream` in `src/backend/main.py`
3. pyreflect pipeline runs and streams logs/results
4. UI consumes SSE events and updates charts in `src/interface/src/components/GraphDisplay.tsx`

**Background Training (Queue):**
1. UI submits job via `POST /api/jobs/submit`
2. Backend enqueues to Redis, triggers Modal worker spawn
3. Modal GPU worker picks up job, runs training with periodic checkpoint saves
4. Worker updates `job.meta` in Redis every second (progress, logs)
5. UI polls `/api/jobs/{job_id}` to show progress
6. On completion, worker uploads model to HuggingFace, saves to MongoDB history

**Pause/Resume:**
1. User clicks Pause in UI
2. Backend sets `pause_requested` flag in job meta
3. Worker sees flag, saves checkpoint to HuggingFace, exits with status "paused"
4. User clicks Resume
5. Backend creates new job, worker loads checkpoint and continues from saved epoch

**History:**
1. UI sends `X-User-ID` header in `src/interface/src/app/page.tsx`
2. Backend stores and fetches history via MongoDB in `src/backend/main.py`

**State Management:**
- Frontend: React state + localStorage (`src/interface/src/app/page.tsx`)
- Backend: Local filesystem under `src/backend/data/` and optional MongoDB
- Job state: Redis (RQ job meta, updated every ~1s by worker)

## Key Abstractions

**Pydantic Models:**
- Purpose: Request/response validation
- Examples: `FilmLayer`, `GenerateRequest`, `GenerateResponse` in `src/backend/main.py`
- Pattern: Pydantic BaseModel

**Streaming Generator:**
- Purpose: SSE progress and log streaming
- Examples: `generate_with_pyreflect_streaming` in `src/backend/main.py`
- Pattern: Generator yielding SSE events

**Checkpoint:**
- Purpose: Serializable training state for persistence
- Examples: `Checkpoint` dataclass in `src/backend/service/services/checkpointing.py`
- Pattern: Dataclass with model/optimizer state dicts, losses, epoch info

**Job Functions:**
- Purpose: Background task execution
- Examples: `run_training_job` in `src/backend/service/jobs/__init__.py`
- Pattern: RQ-compatible function with meta updates and checkpoint integration

**UI Components:**
- Purpose: UI modules and panels
- Examples: `ParameterPanel`, `GraphDisplay`, `ExploreSidebar` in `src/interface/src/components/`
- Pattern: React function components with CSS Modules

## Entry Points

**Next.js App:**
- Location: `src/interface/src/app/page.tsx`
- Triggers: Browser navigation
- Responsibilities: Orchestrates UI state, calls API, renders panels

**NextAuth Route:**
- Location: `src/interface/src/app/api/auth/[...nextauth]/route.ts`
- Triggers: OAuth callbacks and session requests
- Responsibilities: GitHub OAuth, session token enrichment

**FastAPI App:**
- Location: `src/backend/main.py`
- Triggers: HTTP requests to `/api/*`
- Responsibilities: API, streaming generation, uploads, persistence

## Error Handling

**Strategy:**
- Backend raises `HTTPException` for API errors (`src/backend/main.py`)
- Frontend uses try/catch around fetch and logs to console (`src/interface/src/app/page.tsx`, `src/interface/src/components/ExploreSidebar.tsx`)

**Patterns:**
- Backend falls back to default settings when YAML parse fails (`src/backend/main.py`)
- Frontend shows basic alerts for failures in history operations (`src/interface/src/components/ExploreSidebar.tsx`)

## Cross-Cutting Concerns

**Logging:**
- Backend uses `print` and SSE log events (`src/backend/main.py`)
- Frontend logs to console and UI log panel (`src/interface/src/app/page.tsx`)

**Validation:**
- Pydantic model validation in API (`src/backend/main.py`)

**Authentication:**
- NextAuth GitHub OAuth (`src/interface/src/app/api/auth/[...nextauth]/route.ts`)
- Backend authorization via `X-User-ID` header (`src/backend/main.py`)

---

*Architecture analysis: 2026-01-20*
*Update when major patterns change*
