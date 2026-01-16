# Technology Stack

**Analysis Date:** 2026-01-16

## Languages

**Primary:**
- TypeScript 5.x - Frontend source in `src/interface/src/**/*.ts` and `src/interface/src/**/*.tsx` (version in `src/interface/package.json`)
- Python 3.10-3.12 - Backend service in `src/backend/main.py` (version in `src/backend/pyproject.toml` and `src/backend/.python-version`)

**Secondary:**
- CSS - CSS Modules and global styles in `src/interface/src/**/*.css`
- YAML - Backend settings in `src/backend/settings.yml`

## Runtime

**Environment:**
- Node.js (version not pinned) - Required for Next.js (`src/interface/package.json`)
- Python 3.12 - Local dev version (`src/backend/.python-version`)

**Package Manager:**
- npm - `src/interface/package-lock.json`
- bun - `src/interface/bun.lock`
- uv - Python env manager `src/backend/uv.lock`

## Frameworks

**Core:**
- Next.js 16.1.1 - Web UI (`src/interface/package.json`)
- React 19.2.3 - UI runtime (`src/interface/package.json`)
- FastAPI 0.104+ - API server (`src/backend/pyproject.toml`)
- Uvicorn 0.24+ - ASGI server (`src/backend/pyproject.toml`)

**Testing:**
- Not detected (no test dependencies or configs found)

**Build/Dev:**
- TypeScript 5.x - Compiler (`src/interface/package.json`)
- ESLint 9.x - Linting (`src/interface/eslint.config.mjs`)

## Key Dependencies

**Critical:**
- next-auth 4.24.x - GitHub OAuth sessions (`src/interface/package.json`, `src/interface/src/app/api/auth/[...nextauth]/route.ts`)
- recharts 3.x - Data visualization (`src/interface/src/components/GraphDisplay.tsx`)
- html-to-image 1.11.x - PNG export (`src/interface/src/components/GraphDisplay.tsx`)
- pyreflect (git main) - ML pipelines (`src/backend/pyproject.toml`, `src/backend/main.py`)
- numpy 2.1+ - Data processing (`src/backend/pyproject.toml`, `src/backend/main.py`)

**Infrastructure:**
- pymongo 4.6+ - Optional history storage (`src/backend/pyproject.toml`, `src/backend/main.py`)
- huggingface-hub 1.3+ - Model offload (`src/backend/pyproject.toml`, `src/backend/main.py`)
- python-multipart - Upload handling (`src/backend/pyproject.toml`, `src/backend/main.py`)
- python-dotenv - Env loading (`src/backend/pyproject.toml`, `src/backend/main.py`)

## Configuration

**Environment:**
- Backend env vars via `.env` and `.env.example` (`src/backend/.env.example`)
- Frontend env vars via `.env` and `.env.example` (`src/interface/.env.example`)

**Build:**
- `src/interface/next.config.ts` - Next.js config
- `src/interface/tsconfig.json` - TypeScript config
- `src/interface/eslint.config.mjs` - ESLint config
- `src/backend/settings.yml` - Backend runtime settings

## Platform Requirements

**Development:**
- macOS/Linux/Windows with Node.js and Python 3.10-3.12
- uv for Python dependency management (`src/backend/uv.lock`)
- Optional: Bun for frontend installs (`src/interface/bun.lock`)

**Production:**
- Deployment target not detected (assumed Node.js + Python service)
- Backend served via Uvicorn (`src/backend/main.py`)

---

*Stack analysis: 2026-01-16*
*Update after major dependency changes*
