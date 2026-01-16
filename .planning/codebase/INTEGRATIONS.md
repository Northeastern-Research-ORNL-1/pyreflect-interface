# External Integrations

**Analysis Date:** 2026-01-16

## APIs & External Services

**Payment Processing:**
- Not detected

**Email/SMS:**
- Not detected

**External APIs:**
- GitHub OAuth - Sign-in via NextAuth (`src/interface/src/app/api/auth/[...nextauth]/route.ts`)
  - Integration method: NextAuth GitHubProvider
  - Auth: `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` in `src/interface/.env.example`
- Hugging Face Hub - Model storage and download (`src/backend/main.py`)
  - Integration method: `huggingface_hub.HfApi`
  - Auth: `HF_TOKEN`, `HF_REPO_ID` in `src/backend/.env.example`

## Data Storage

**Databases:**
- MongoDB (optional) - Persist generation history (`src/backend/main.py`)
  - Connection: `MONGODB_URI` in `src/backend/.env.example`
  - Client: `pymongo` in `src/backend/pyproject.toml`
  - Collections: `PyReflect.generations` in `src/backend/main.py`

**File Storage:**
- Local filesystem - Datasets and models in `src/backend/data/`
  - Models: `src/backend/data/models/`
  - Curves: `src/backend/data/curves/`
  - Experimental curves: `src/backend/data/curves/expt/`
- Hugging Face dataset (optional) - Remote model files (`src/backend/main.py`)

**Caching:**
- Not detected

## Authentication & Identity

**Auth Provider:**
- NextAuth (GitHub) - UI session (`src/interface/src/app/api/auth/[...nextauth]/route.ts`)
  - Token storage: NextAuth session cookie (managed by NextAuth)
  - Session usage: GitHub user id is passed to API via `X-User-ID` header in `src/interface/src/app/page.tsx`

**OAuth Integrations:**
- GitHub OAuth - Credentials in `src/interface/.env.example`

## Monitoring & Observability

**Error Tracking:**
- Not detected

**Analytics:**
- Not detected

**Logs:**
- Console logs only (FastAPI uses `print` in `src/backend/main.py`, frontend uses `console` in `src/interface/src`)

## CI/CD & Deployment

**Hosting:**
- Not detected (README describes local dev only)

**CI Pipeline:**
- Not detected (no workflows in repo)

## Environment Configuration

**Development:**
- Backend env vars: `PRODUCTION`, `CORS_ORIGINS`, `MONGODB_URI`, `HF_TOKEN`, `HF_REPO_ID` (`src/backend/.env.example`)
- Frontend env vars: `NEXT_PUBLIC_API_URL`, `NEXTAUTH_URL`, `NEXTAUTH_SECRET`, `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET` (`src/interface/.env.example`)

**Staging:**
- Not detected

**Production:**
- Not detected (env var names listed above)

## Webhooks & Callbacks

**Incoming:**
- Not detected

**Outgoing:**
- Not detected

---

*Integration audit: 2026-01-16*
*Update when adding/removing external services*
