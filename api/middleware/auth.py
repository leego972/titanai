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
        return None

    if not config.API_KEY:
        return None

    # Check Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token == config.API_KEY or check_identity(token)[0]:
            return token
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Invalid API key", "type": "auth_error", "code": "invalid_api_key"}},
        )

    # Check X-API-Key header
    x_api_key = request.headers.get("X-API-Key", "").strip()
    if x_api_key:
        if x_api_key == config.API_KEY or check_identity(x_api_key)[0]:
            return x_api_key
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"message": "Invalid API key", "type": "auth_error", "code": "invalid_api_key"}},
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={"error": {"message": "API key required. Use Authorization: Bearer <key> or X-API-Key header.", "type": "auth_error", "code": "missing_api_key"}},
    )

def is_request_private(headers: dict, api_key: Optional[str]) -> bool:
    """
    Determine persona based on origin domain:
      archibaldtitan.com  → always Private/Cyber (Expert mode)
      virelle.life        → always Public/Compliant mode
      zippyfixer.com      → always Public/Compliant mode
      direct API access   → Private if valid API key supplied, else Public
    """
    origin  = headers.get("origin",  "").lower()
    referer = headers.get("referer", "").lower()
    source  = origin + referer

    # archibaldtitan.com always gets the cyber/expert persona
    if "archibaldtitan.com" in source:
        return True

    # Public-facing sites always get the compliant persona
    public_domains = ["virelle.life", "zippyfixer.com"]
    for domain in public_domains:
        if domain in source:
            return False

    # Direct API access: private if a valid key was supplied
    if api_key:
        return True

    # Default: public/safe
    return False
