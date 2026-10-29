import contextlib
import json
import re
from typing import Any

from tokenwatch.models import PromptBreakdown
from tokenwatch.security import DataRedactor

# 100% offline token estimation regex compatible with standard Python re
OFFLINE_TOKEN_REGEX = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]{1,3}|[^\s\w]+|\s+""", re.IGNORECASE)


class PromptSlotDecomposer:
    """
    Analyzes and decomposes raw chat payload components into granular token slots.
    Designed for 100% private, offline operation with zero network dependencies.
    """

    def __init__(self, default_encoding: str = "cl100k_base"):
        self.encoder = None
        # Attempt to use tiktoken ONLY if locally available without network fetching
        try:
            import tiktoken

            self.encoder = tiktoken.get_encoding(default_encoding)
        except Exception:  # noqa: BLE001
            self.encoder = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder:
            with contextlib.suppress(Exception):
                return len(self.encoder.encode(text))

        # 100% offline regex token counter
        tokens = OFFLINE_TOKEN_REGEX.findall(text)
        if tokens:
            return len(tokens)
        return max(1, len(text) // 4)

    def decompose(self, payload: dict[str, Any], redact_secrets: bool = True) -> PromptBreakdown:
        breakdown = PromptBreakdown()

        # 1. Measure Tools / Function calling schemas
        tools = payload.get("tools", [])
        if tools:
            tools_str = json.dumps(tools)
            if redact_secrets:
                tools_str = DataRedactor.redact_text(tools_str)
            breakdown.tool_schemas = self.count(tools_str)

        # 2. Iterate through messages
        messages: list[dict[str, Any]] = payload.get("messages", [])
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break

        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            raw_content = msg.get("content", "")
            content = json.dumps(raw_content) if isinstance(raw_content, (list, dict)) else str(raw_content or "")
            if redact_secrets:
                content = DataRedactor.redact_text(content)
            tokens = self.count(content)

            if role == "system":
                breakdown.system_instructions += tokens
            elif idx == last_user_idx:
                # Most recent user message is the active task payload
                breakdown.task_payload += tokens
            else:
                # Detect telemetry dumps or IoT sensor patterns
                lower_content = content.lower()
                telemetry_markers = ["sensor", "temperature", "humidity", "state:", "entity_id", "telemetry", "metric:"]
                if any(k in lower_content for k in telemetry_markers):
                    breakdown.iot_telemetry += tokens
                else:
                    breakdown.conversation_history += tokens

        breakdown.total_prompt_tokens = (
            breakdown.system_instructions
            + breakdown.tool_schemas
            + breakdown.conversation_history
            + breakdown.iot_telemetry
            + breakdown.task_payload
        )
        return breakdown

