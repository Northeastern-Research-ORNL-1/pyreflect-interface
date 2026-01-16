# Repository Guidelines

## Project Structure & Module Organization
- `src/backend/` hosts the FastAPI service (`src/backend/main.py`) plus runtime data under `src/backend/data/`.
- `src/interface/` is the Next.js frontend. UI lives in `src/interface/src/app/` and reusable components in `src/interface/src/components/`.
- Static assets live in `src/interface/public/`.
- Planning artifacts (if used) live under `.planning/`.

## Build, Test, and Development Commands
Frontend (Next.js):
```bash
cd src/interface
bun install            # or: npm install
bun run dev            # local dev server (http://localhost:3000)
bun run build          # production build
bun run start          # serve production build
bun run lint           # ESLint
```

Backend (FastAPI):
```bash
cd src/backend
uv python pin 3.12
uv sync
cp .env.example .env
uv run uvicorn main:app --reload --port 8000
```

## Coding Style & Naming Conventions
- TypeScript/React: 2-space indentation, semicolons, PascalCase component files (`GraphDisplay.tsx`), CSS Modules as `*.module.css`.
- Next.js App Router uses `page.tsx` and `layout.tsx` under `src/interface/src/app/`.
- Imports use the `@/` alias for `src/interface/src/` (`src/interface/tsconfig.json`).
- Python uses 4-space indentation, snake_case functions, and Pydantic models in PascalCase (`src/backend/main.py`).
- Linting: ESLint config in `src/interface/eslint.config.mjs`.

## Testing Guidelines
- No automated tests are configured currently (no test scripts or configs detected).
- If you add tests, document the framework and naming conventions here.

## Commit & Pull Request Guidelines
- Commit history generally follows conventional prefixes: `feat:`, `fix:`, `chore:`. Keep messages short and imperative.
- PRs should include a clear summary, note any API or config changes, and add screenshots for UI changes.

## Security & Configuration Tips
- Do not commit secrets. Add new variables to `.env.example` in both `src/backend/` and `src/interface/`.
- Runtime data under `src/backend/data/` should remain untracked.
