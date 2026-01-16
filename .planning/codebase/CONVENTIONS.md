# Coding Conventions

**Analysis Date:** 2026-01-16

## Naming Patterns

**Files:**
- PascalCase for React components (`src/interface/src/components/ParameterPanel.tsx`)
- *.module.css for CSS Modules (`src/interface/src/components/ParameterPanel.module.css`)
- page.tsx/layout.tsx for Next.js App Router (`src/interface/src/app/page.tsx`, `src/interface/src/app/layout.tsx`)

**Functions:**
- camelCase for functions in TS (`src/interface/src/app/page.tsx`)
- snake_case for functions in Python (`src/backend/main.py`)
- Handlers use handle* prefix (`src/interface/src/components/ExploreSidebar.tsx`)

**Variables:**
- camelCase for variables (`src/interface/src/app/page.tsx`)
- UPPER_SNAKE_CASE for constants (`src/interface/src/app/page.tsx`, `src/backend/main.py`)
- No private prefix conventions observed

**Types:**
- PascalCase for interfaces and type aliases (`src/interface/src/types/index.ts`)
- Pydantic models in PascalCase (`src/backend/main.py`)

## Code Style

**Formatting:**
- 2-space indentation in TS/TSX (`src/interface/src/app/page.tsx`)
- 4-space indentation in Python (`src/backend/main.py`)
- Semicolons present in TS/TSX
- Quotes are mixed: single quotes in most components (`src/interface/src/app/page.tsx`), double quotes in config/auth files (`src/interface/src/app/api/auth/[...nextauth]/route.ts`, `src/interface/next.config.ts`)

**Linting:**
- ESLint with Next.js config (`src/interface/eslint.config.mjs`)
- Run: `npm run lint` or `bun run lint` (script in `src/interface/package.json`)

## Import Organization

**Order:**
1. External packages (`react`, `next-auth`) in `src/interface/src/app/page.tsx`
2. Internal modules using alias (`@/types`) in `src/interface/src/app/page.tsx`
3. Relative imports for local components (`../components/ParameterPanel`)
4. Type-only imports using `type` (`src/interface/src/components/ParameterPanel.tsx`)

**Grouping:**
- Blank lines between groups are common in components
- No enforced sorting detected

**Path Aliases:**
- `@/` maps to `src/interface/src` (`src/interface/tsconfig.json`)

## Error Handling

**Patterns:**
- Backend raises `HTTPException` for API failures (`src/backend/main.py`)
- Frontend uses try/catch around fetch and logs errors (`src/interface/src/app/page.tsx`, `src/interface/src/components/ExploreSidebar.tsx`)

**Error Types:**
- API endpoints return FastAPI HTTP error responses (`src/backend/main.py`)
- UI alerts for some failures (history operations) (`src/interface/src/components/ExploreSidebar.tsx`)

## Logging

**Framework:**
- Backend uses `print` statements (`src/backend/main.py`)
- Frontend uses `console.error` and UI log list (`src/interface/src/app/page.tsx`)

**Patterns:**
- SSE log events from backend to UI (`src/backend/main.py`, `src/interface/src/app/page.tsx`)

## Comments

**When to Comment:**
- Inline comments for behavior notes and compatibility (`src/interface/src/components/GraphDisplay.tsx`)
- Docstrings for Python models and functions (`src/backend/main.py`)

**TODO Comments:**
- Not detected

## Function Design

**Size:**
- Some large functions and components (notably `src/interface/src/app/page.tsx`, `src/backend/main.py`)

**Parameters:**
- React components receive props objects with typed interfaces (`src/interface/src/components/ParameterPanel.tsx`)

**Return Values:**
- Explicit return in React components and API handlers

## Module Design

**Exports:**
- Default exports for React components (`src/interface/src/components/*.tsx`)
- Named exports for types (`src/interface/src/types/index.ts`)

**Barrel Files:**
- Type exports consolidated in `src/interface/src/types/index.ts`

---

*Convention analysis: 2026-01-16*
*Update when patterns change*
