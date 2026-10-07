import os
import re
from typing import Dict, Any, List, Optional
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER_NAME = "X-TokenWatch-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

def get_configured_api_key() -> Optional[str]:
    return os.environ.get("TOKENWATCH_API_KEY", "").strip() or None

async def verify_api_key(request: Request) -> bool:
    """
    Enforces pre-shared API key authentication if TOKENWATCH_API_KEY is configured.
    If no key is configured in the environment, strictly limits access to localhost.
    """
    configured_key = get_configured_api_key()
    if not configured_key:
        # Fallback to localhost-only access for secure-by-default behavior
        client_host = request.client.host if request.client else ""
        if client_host in ("127.0.0.1", "::1", "localhost"):
            return True
        raise HTTPException(status_code=401, detail="Unauthorized: TOKENWATCH_API_KEY is not configured and request is not from localhost")

    header_key = request.headers.get(API_KEY_HEADER_NAME)
    auth_header = request.headers.get("Authorization")
    
    if header_key and header_key == configured_key:
        return True

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token == configured_key:
            return True

    raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-TokenWatch-Key")

class DataRedactor:
    """
    Redacts sensitive home network secrets, credentials, alarm PINs, and PII
    from prompt contexts and telemetry records before storage.
    """
    PATTERNS = [
        # Cloud LLM and Git API Keys
        (re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.IGNORECASE), "[REDACTED_API_KEY]"),
        (re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}", re.IGNORECASE), "[REDACTED_ANTHROPIC_KEY]"),
        (re.compile(r"AIza[0-9A-Za-z-_]{35}"), "[REDACTED_GOOGLE_KEY]"),
        (re.compile(r"ghp_[0-9a-zA-Z]{36}"), "[REDACTED_GITHUB_KEY]"),
        
        # Wi-Fi passwords, Alarm PINs, Secrets
        (re.compile(r"(password|passwd|pwd|pass)\s*[:=]\s*['\"]?([^'\"\s,]+)['\"]?", re.IGNORECASE), r"\1: [REDACTED_PASSWORD]"),
        (re.compile(r"(alarm_code|pin|security_code)\s*[:=]\s*['\"]?(\d{4,8})['\"]?", re.IGNORECASE), r"\1: [REDACTED_PIN]"),
        (re.compile(r"(wifi_pass|wifi_password|wpa_key)\s*[:=]\s*['\"]?([^'\"\s,]+)['\"]?", re.IGNORECASE), r"\1: [REDACTED_WIFI_PASS]"),
        
        # Bearer tokens & Private keys
        (re.compile(r"Bearer\s+[a-zA-Z0-9._-]{20,}", re.IGNORECASE), "Bearer [REDACTED_TOKEN]"),
        (re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]")
    ]

    @classmethod
    def redact_text(cls, text: str) -> str:
        if not text:
            return text
        redacted = text
        for pattern, replacement in cls.PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted

    @classmethod
    def sanitize_dict(cls, data: Any) -> Any:
        if isinstance(data, str):
            return cls.redact_text(data)
        elif isinstance(data, dict):
            return {k: cls.sanitize_dict(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.sanitize_dict(item) for item in data]
        return data
