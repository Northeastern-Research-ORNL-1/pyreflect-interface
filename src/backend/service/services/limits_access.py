from __future__ import annotations

from typing import Sequence

from ..config import IS_PRODUCTION, LIMITS_WHITELIST_USER_IDS, LOCAL_LIMITS, PRODUCTION_LIMITS


def is_whitelisted_user(*, user_id: str | None, whitelist: Sequence[str] | None = None) -> bool:
    if not user_id:
        return False

    normalized = user_id.strip()
    if not normalized:
        return False

    whitelist_ids = list(whitelist if whitelist is not None else LIMITS_WHITELIST_USER_IDS)
    return normalized in whitelist_ids


def get_effective_limits(
    *,
    user_id: str | None,
) -> tuple[dict[str, int | float], bool, str]:
    """
    Returns (limits, access_granted, limit_source).

    limit_source:
    - local_dev: running with PRODUCTION=false
    - whitelist: PRODUCTION=true and user is in LIMITS_WHITELIST_USER_IDS
    - production: PRODUCTION=true and user is not whitelisted
    """

    if not IS_PRODUCTION:
        return dict(LOCAL_LIMITS), True, "local_dev"

    if is_whitelisted_user(user_id=user_id):
        return dict(LOCAL_LIMITS), True, "whitelist"

    return dict(PRODUCTION_LIMITS), False, "production"
