from __future__ import annotations

import secrets

from fastapi import HTTPException


def require_user_id(*, is_production: bool, x_user_id: str | None) -> None:
    if is_production and not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-ID required")


def require_admin_token(
    *,
    is_production: bool,
    admin_token: str | None,
    x_admin_token: str | None,
) -> None:
    if not is_production:
        return
    if not admin_token:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN not configured")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, admin_token):
        raise HTTPException(status_code=401, detail="Invalid admin token")
