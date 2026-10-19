import hmac
import os
import re
from pathlib import Path
from typing import Any, ClassVar

from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader

API_KEY_HEADER_NAME = "X-TokenWatch-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

_DOTENV_LOADED = False


def load_dotenv_if_present(dotenv_path: str | Path | None = None) -> None:
    """
    Zero-dependency .env file loader. Populates os.environ with key-value pairs
    from .env if not already set in the process environment.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED and dotenv_path is None:
        return

    target = Path(dotenv_path) if dotenv_path else Path.cwd() / ".env"
    if target.is_file():
        try:
            for raw_line in target.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            pass
    if dotenv_path is None:
        _DOTENV_LOADED = True


def get_configured_api_key() -> str | None:
    load_dotenv_if_present()
    return os.environ.get("TOKENWATCH_API_KEY", "").strip() or None


async def verify_api_key(request: Request) -> bool:
    """
    Enforces pre-shared API key authentication if TOKENWATCH_API_KEY is configured.
    If no key is configured in the environment, strictly limits access to localhost.
    Uses constant-time comparison to prevent timing attacks.
    """
    configured_key = get_configured_api_key()
    if not configured_key:
        # Fallback to localhost-only access for secure-by-default behavior
        client_host = request.client.host if request.client else ""
        if client_host in ("127.0.0.1", "::1", "localhost", "testclient"):
            return True
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: TOKENWATCH_API_KEY is not configured and request is not from localhost",
        )

    header_key = request.headers.get(API_KEY_HEADER_NAME)
    auth_header = request.headers.get("Authorization")

    if header_key and hmac.compare_digest(header_key, configured_key):
        return True

    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if hmac.compare_digest(token, configured_key):
            return True

    raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-TokenWatch-Key")


class DataRedactor:
    """
    Redacts sensitive home network secrets, credentials, alarm PINs, and PII
    from prompt contexts and telemetry records before storage.
    """

    PATTERNS: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        # Cloud LLM and Git API Keys (specific prefixes before generic sk-)
        (re.compile(r"sk-ant-[a-zA-Z0-9_-]{20,}", re.IGNORECASE), "[REDACTED_ANTHROPIC_KEY]"),
        (re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.IGNORECASE), "[REDACTED_API_KEY]"),
        (re.compile(r"AIza[0-9A-Za-z-_]{35}"), "[REDACTED_GOOGLE_KEY]"),
        (re.compile(r"ghp_[0-9a-zA-Z]{36}"), "[REDACTED_GITHUB_KEY]"),
        (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
        (re.compile(r"hf_[a-zA-Z0-9]{34}"), "[REDACTED_HF_KEY]"),
        (re.compile(r"xox[baprs]-[0-9a-zA-Z-]{10,48}"), "[REDACTED_SLACK_KEY]"),
        # Wi-Fi passwords, Alarm PINs, Secrets
        (
            re.compile(r"(password|passwd|pwd|pass)\s*[:=]\s*['\"]?([^'\"\s,]+)['\"]?", re.IGNORECASE),
            r"\1: [REDACTED_PASSWORD]",
        ),
        (
            re.compile(r"(alarm_code|pin|security_code)\s*[:=]\s*['\"]?(\d{4,8})['\"]?", re.IGNORECASE),
            r"\1: [REDACTED_PIN]",
        ),
        (
            re.compile(r"(wifi_pass|wifi_password|wpa_key)\s*[:=]\s*['\"]?([^'\"\s,]+)['\"]?", re.IGNORECASE),
            r"\1: [REDACTED_WIFI_PASS]",
        ),
        # Bearer tokens & Private keys
        (re.compile(r"Bearer\s+[a-zA-Z0-9._-]{20,}", re.IGNORECASE), "Bearer [REDACTED_TOKEN]"),
        (
            re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----"),
            "[REDACTED_PRIVATE_KEY]",
        ),
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
        if isinstance(data, dict):
            return {k: cls.sanitize_dict(v) for k, v in data.items()}
        if isinstance(data, list):
            return [cls.sanitize_dict(item) for item in data]
        return data

