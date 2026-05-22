"""
TitanAI API — Authentication Middleware
=========================================
Simple API key authentication via Bearer token or X-API-Key header.
Auth is optional in dev mode (TITAN_REQUIRE_AUTH=false).
"""
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

from ..core.config import config
from ..core.identity import check_identity

security = HTTPBearer(auto_error=False)

async def verify_api_key(request: Request) -> Optional[str]:
    """
    Verify API key from Authorization: Bearer <key> or X-API-Key header.
    Returns the key if valid, None if auth is disabled.
    Raises 401 if auth is required and key is missing/invalid.
    """
    if not config.REQUIRE_AUTH:
        return None  # Auth disabled — allow all requests

    if not config.API_KEY:
        return None  # No key configured — allow all (misconfiguration warning logged at startup)

    # Check Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token == config.API_KEY or check_identity(token):
            return token
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Invalid API key", "type": "auth_error", "code": "invalid_api_key"}},
        )

    # Check X-API-Key header
    x_api_key = request.headers.get("X-API-Key", "").strip()
    if x_api_key:
        if x_api_key == config.API_KEY or check_identity(x_api_key):
            return x_api_key
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Invalid API key", "type": "auth_error", "code": "invalid_api_key"}},
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": {"message": "API key required. Use Authorization: Bearer <key> or X-API-Key header.", "type": "auth_error", "code": "missing_api_key"}},
    )
