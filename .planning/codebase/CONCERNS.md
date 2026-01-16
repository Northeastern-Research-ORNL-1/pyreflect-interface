# Codebase Concerns

**Analysis Date:** 2026-01-16

## Tech Debt

**Monolithic backend module:**
- Issue: API routes, data processing, streaming, storage, and config live in one file
- Why: Single-file prototype grew as features were added
- Impact: Hard to test and refactor; higher risk of regressions
- Fix approach: Split `src/backend/main.py` into modules (api, services, storage, config)

**Large UI page component:**
- Issue: Most UI state, API calls, and side effects are in one file
- Why: Centralized initial implementation
- Impact: Hard to reason about changes and reuse logic
- Fix approach: Extract hooks and service helpers from `src/interface/src/app/page.tsx`

## Known Bugs

**Duplicate console log entries:**
- Symptoms: Each log line appears twice in the UI console
- Trigger: Any call to `addLog` in `src/interface/src/app/page.tsx`
- Workaround: None
- Root cause: Duplicate `setConsoleLogs` calls in `addLog`

## Security Considerations

**Unverified user identity in backend:**
- Risk: Any client can spoof `X-User-ID` to read or delete history
- Current mitigation: None (trusts header from client)
- Recommendations: Verify NextAuth session or use signed JWTs in `src/backend/main.py` for `/api/history` endpoints

## Performance Bottlenecks

**Long-running training in request thread:**
- Problem: CPU/GPU-heavy training runs inline in API requests
- Measurement: Not measured
- Cause: Synchronous training in `generate_with_pyreflect*` in `src/backend/main.py`
- Improvement path: Offload to background worker or task queue; return job IDs

## Fragile Areas

**Upload role mapping and settings updates:**
- Why fragile: Role inference and settings file updates are tightly coupled in one map
- Common failures: New upload role without mapping breaks settings updates
- Safe modification: Update `UPLOAD_ROLE_SETTINGS_MAP` and `apply_upload_to_settings` together in `src/backend/main.py`
- Test coverage: No tests for upload role handling

## Scaling Limits

**Production parameter caps:**
- Current capacity: Limits enforced via `LIMITS` in `src/backend/main.py`
- Limit: `max_curves=5000`, `max_film_layers=10`, `max_epochs=50`, `max_batch_size=64` (production)
- Symptoms at limit: 400 responses from `/api/generate*`
- Scaling path: Increase env vars in `src/backend/.env.example` or add background scaling

## Dependencies at Risk

**pyreflect pinned to main branch:**
- Risk: Upstream breaking changes on `main`
- Impact: Backend failures during install or runtime
- Migration plan: Pin to a tag or commit in `src/backend/pyproject.toml`

## Missing Critical Features

**Not detected**

## Test Coverage Gaps

**No automated tests:**
- What's not tested: Backend endpoints and frontend UI logic
- Risk: Regressions in generation, uploads, and history flows
- Priority: High
- Difficulty to test: Medium (needs API and UI test setup)

---

*Concerns audit: 2026-01-16*
*Update as issues are fixed or new ones discovered*
