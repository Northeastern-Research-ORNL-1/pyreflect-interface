from __future__ import annotations

import pytest
from fastapi import HTTPException


def test_require_user_id_allows_dev() -> None:
    from service.services.guards import require_user_id

    require_user_id(is_production=False, x_user_id=None)


def test_require_user_id_blocks_prod() -> None:
    from service.services.guards import require_user_id

    with pytest.raises(HTTPException) as exc:
        require_user_id(is_production=True, x_user_id=None)
    assert exc.value.status_code == 401


def test_require_admin_token_allows_dev() -> None:
    from service.services.guards import require_admin_token

    require_admin_token(is_production=False, admin_token=None, x_admin_token=None)


def test_require_admin_token_requires_configured() -> None:
    from service.services.guards import require_admin_token

    with pytest.raises(HTTPException) as exc:
        require_admin_token(is_production=True, admin_token=None, x_admin_token=None)
    assert exc.value.status_code == 503


def test_require_admin_token_requires_match() -> None:
    from service.services.guards import require_admin_token

    with pytest.raises(HTTPException) as exc:
        require_admin_token(is_production=True, admin_token="t", x_admin_token=None)
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException) as exc:
        require_admin_token(is_production=True, admin_token="t", x_admin_token="wrong")
    assert exc.value.status_code == 401

    require_admin_token(is_production=True, admin_token="t", x_admin_token="t")
