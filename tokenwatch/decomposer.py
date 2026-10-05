import json
from typing import Dict, Any, List
import tiktoken
from tokenwatch.models import PromptBreakdown

class PromptSlotDecomposer:
    """
    Analyzes and decomposes raw chat payload components into granular token slots.
    """
    def __init__(self, default_encoding: str = "cl100k_base"):
        try:
            self.encoder = tiktoken.get_encoding(default_encoding)
        except Exception:
            self.encoder = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception:
                pass
        # Fallback approximation (~4 chars per token)
        return max(1, len(text) // 4)

    def decompose(self, payload: Dict[str, Any]) -> PromptBreakdown:
        breakdown = PromptBreakdown()
        
        # 1. Measure Tools / Function calling schemas
        tools = payload.get("tools", [])
        if tools:
            tools_str = json.dumps(tools)
            breakdown.tool_schemas = self.count(tools_str)

        # 2. Iterate through messages
        messages: List[Dict[str, Any]] = payload.get("messages", [])
        total_msgs = len(messages)
        
        for idx, msg in enumerate(messages):
            role = msg.get("role", "")
            content = str(msg.get("content", ""))
            tokens = self.count(content)

            if role == "system":
                breakdown.system_instructions += tokens
            elif idx == total_msgs - 1 and role == "user":
                # Final user message is the active task payload
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
