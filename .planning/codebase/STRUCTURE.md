# Codebase Structure

**Analysis Date:** 2026-01-16

## Directory Layout

```
pyreflect-interface/
├── .planning/                 # Planning artifacts (GSD)
│   └── codebase/              # Codebase analysis docs
├── README.md                  # Project overview and setup
├── package.json               # Root manifest (empty)
├── package-lock.json          # Root lockfile
├── plan.md                    # Local planning notes
└── src/
    ├── backend/               # FastAPI backend
    │   ├── main.py            # API server and core logic
    │   ├── pyproject.toml     # Python dependencies
    │   ├── uv.lock            # uv lockfile
    │   ├── settings.yml       # Backend settings
    │   ├── data/              # Runtime data and models
    │   │   ├── curves/        # Curve data uploads
    │   │   └── models/        # Saved model weights
    │   └── .env.example       # Backend env template
    └── interface/             # Next.js frontend
        ├── package.json       # Frontend dependencies
        ├── tsconfig.json      # TypeScript config
        ├── next.config.ts     # Next.js config
        ├── eslint.config.mjs  # ESLint config
        ├── public/            # Static assets
        └── src/               # Frontend source
            ├── app/           # App Router pages and routes
            ├── components/    # UI components
            └── types/         # Shared types
```

## Directory Purposes

**src/backend/**
- Purpose: FastAPI API server and ML orchestration
- Contains: API routes, model training/inference, file handling
- Key files: `src/backend/main.py`, `src/backend/pyproject.toml`
- Subdirectories: `src/backend/data/` for runtime data

**src/backend/data/**
- Purpose: Runtime datasets and model artifacts
- Contains: `.npy` curve files and `.pth` models
- Key files: `src/backend/data/curves/`, `src/backend/data/models/`
- Subdirectories: `src/backend/data/curves/expt/` for experimental curves

**src/interface/**
- Purpose: Next.js frontend application
- Contains: App Router pages, UI components, assets
- Key files: `src/interface/src/app/page.tsx`, `src/interface/src/app/layout.tsx`
- Subdirectories: `src/interface/src/components/`, `src/interface/src/types/`, `src/interface/public/`

**src/interface/src/app/**
- Purpose: App Router pages and API routes
- Contains: `page.tsx`, `layout.tsx`, `api/auth/[...nextauth]/route.ts`
- Subdirectories: `src/interface/src/app/api/`

**src/interface/src/components/**
- Purpose: UI components and panels
- Contains: `ParameterPanel.tsx`, `GraphDisplay.tsx`, `ExploreSidebar.tsx`
- Subdirectories: None (flat component list)

**src/interface/src/types/**
- Purpose: Shared TypeScript types
- Contains: `index.ts` with exported interfaces
- Subdirectories: None

## Key File Locations

**Entry Points:**
- `src/interface/src/app/page.tsx` - Main UI page
- `src/backend/main.py` - FastAPI app and routes

**Configuration:**
- `src/interface/tsconfig.json` - TypeScript config
- `src/interface/next.config.ts` - Next.js config
- `src/interface/eslint.config.mjs` - ESLint config
- `src/backend/settings.yml` - Backend runtime settings
- `src/backend/pyproject.toml` - Python dependencies

**Core Logic:**
- `src/backend/main.py` - API routes, ML pipeline orchestration, storage
- `src/interface/src/components/ParameterPanel.tsx` - Input controls and uploads
- `src/interface/src/components/GraphDisplay.tsx` - Chart rendering

**Testing:**
- Not detected (no test directory or config files)

**Documentation:**
- `README.md` - User and developer instructions

## Naming Conventions

**Files:**
- PascalCase.tsx for React components (`src/interface/src/components/ParameterPanel.tsx`)
- *.module.css for CSS Modules (`src/interface/src/components/ParameterPanel.module.css`)
- page.tsx/layout.tsx for App Router (`src/interface/src/app/page.tsx`)

**Directories:**
- Lowercase for app structure (`src/interface/src/app`, `src/interface/src/components`)
- Feature grouping by folder (backend vs interface)

**Special Patterns:**
- `index.ts` for type exports (`src/interface/src/types/index.ts`)
- `[...nextauth]` for NextAuth catch-all route (`src/interface/src/app/api/auth/[...nextauth]/route.ts`)

## Where to Add New Code

**New Feature:**
- Primary code: `src/interface/src/components/` or `src/interface/src/app/`
- Tests: Not detected (no test pattern defined)
- Config if needed: `src/interface/next.config.ts` or `src/backend/settings.yml`

**New Component/Module:**
- Implementation: `src/interface/src/components/`
- Types: `src/interface/src/types/index.ts`
- Styles: `src/interface/src/components/*.module.css`

**New Route/Command:**
- Definition: `src/backend/main.py` (FastAPI routes)
- Handler: `src/backend/main.py`
- Tests: Not detected

**Utilities:**
- Shared helpers: Add within `src/interface/src/components/` or extract a new `src/interface/src/lib/` (not present yet)
- Type definitions: `src/interface/src/types/index.ts`

## Special Directories

**src/backend/data/**
- Purpose: Runtime datasets and models
- Source: Uploaded by API (`/api/upload` in `src/backend/main.py`)
- Committed: Should be gitignored (contains generated data)

**src/interface/.next/**
- Purpose: Next.js build output
- Source: Generated by Next.js
- Committed: No (build artifacts)

**src/interface/node_modules/**
- Purpose: Frontend dependencies
- Source: npm or bun install
- Committed: No

**src/backend/.venv/**
- Purpose: Local Python virtual environment
- Source: uv
- Committed: No

**src/backend/__pycache__/**
- Purpose: Python bytecode cache
- Source: Python runtime
- Committed: No

---

*Structure analysis: 2026-01-16*
*Update when directory structure changes*
