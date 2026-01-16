# Architecture

**Analysis Date:** 2026-01-16

## Pattern Overview

**Overall:** Monorepo with Next.js frontend and FastAPI backend (client-server architecture)

**Key Characteristics:**
- Next.js App Router UI with client components (`src/interface/src/app`, `src/interface/src/components`)
- FastAPI backend in a single module (`src/backend/main.py`)
- Server-Sent Events for long-running generation (`/api/generate/stream` in `src/backend/main.py`)
- Optional persistence to MongoDB and Hugging Face (`src/backend/main.py`)

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
- Purpose: HTTP endpoints for generation, uploads, and history
- Contains: FastAPI routes and response models
- Depends on: pyreflect pipelines, storage adapters
- Used by: UI via REST and SSE
- Location: `src/backend/main.py`

**Processing Layer:**
- Purpose: Curve generation, training, inference pipelines
- Contains: pyreflect pipelines, numpy processing, streaming output
- Depends on: pyreflect, numpy, torch
- Used by: API endpoints
- Location: `src/backend/main.py`

**Persistence Layer:**
- Purpose: Optional history storage and model persistence
- Contains: MongoDB access, local filesystem storage, Hugging Face offload
- Depends on: pymongo, filesystem, huggingface_hub
- Used by: API endpoints
- Location: `src/backend/main.py`, `src/backend/data/`

## Data Flow

**Generation (SSE):**
1. UI submits request from `src/interface/src/app/page.tsx`
2. Backend handles `POST /api/generate/stream` in `src/backend/main.py`
3. pyreflect pipeline runs and streams logs/results
4. UI consumes SSE events and updates charts in `src/interface/src/components/GraphDisplay.tsx`

**History:**
1. UI sends `X-User-ID` header in `src/interface/src/app/page.tsx`
2. Backend stores and fetches history via MongoDB in `src/backend/main.py`

**State Management:**
- Frontend: React state + localStorage (`src/interface/src/app/page.tsx`)
- Backend: Local filesystem under `src/backend/data/` and optional MongoDB

## Key Abstractions

**Pydantic Models:**
- Purpose: Request/response validation
- Examples: `FilmLayer`, `GenerateRequest`, `GenerateResponse` in `src/backend/main.py`
- Pattern: Pydantic BaseModel

**Streaming Generator:**
- Purpose: SSE progress and log streaming
- Examples: `generate_with_pyreflect_streaming` in `src/backend/main.py`
- Pattern: Generator yielding SSE events

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

*Architecture analysis: 2026-01-16*
*Update when major patterns change*
