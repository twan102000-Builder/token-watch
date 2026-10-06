import re
from typing import List, Dict, Any, Tuple, Optional

class DynamicToolPruner:
    """
    Ranks tools against the user prompt using token-frequency scoring
    and prunes non-relevant tools to dramatically reduce schema bloat.
    """
    def __init__(self, min_keep: int = 3, max_keep: int = 8):
        self.min_keep = min_keep
        self.max_keep = max_keep

    def _tokenize(self, text: str) -> set:
        return set(re.findall(r"\w+", text.lower()))

    def score_tool(self, tool: Dict[str, Any], query_tokens: set) -> float:
        function = tool.get("function", {})
        name = function.get("name", "")
        desc = function.get("description", "")
        params = str(function.get("parameters", {}))
        
        target_text = f"{name} {name.replace('_', ' ')} {desc} {params}".lower()
        target_tokens = self._tokenize(target_text)
        
        if not query_tokens or not target_tokens:
            return 0.0

        matches = query_tokens.intersection(target_tokens)
        return len(matches) / (len(query_tokens) ** 0.5)

    def prune_tools(self, tools: List[Dict[str, Any]], user_query: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Returns (kept_tools, pruned_tool_names)
        """
        if len(tools) <= self.min_keep:
            return tools, []

        query_tokens = self._tokenize(user_query)
        scored = []
        for t in tools:
            score = self.score_tool(t, query_tokens)
            name = t.get("function", {}).get("name", "unknown")
            scored.append((score, name, t))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top max_keep
        kept = [item[2] for item in scored[:self.max_keep]]
        pruned_names = [item[1] for item in scored[self.max_keep:]]

        return kept, pruned_names

class PrefixCacheStabilizer:
    """
    Reorders chat message components so static blocks (system prompt, schemas)
    sit at the prefix to maximize KV cache hit rates.
    """
    @staticmethod
    def stabilize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = payload.get("messages", [])
        if not messages:
            return payload

        # Extract system messages vs others
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        # Ensure system messages are strictly ordered first
        payload["messages"] = system_msgs + other_msgs
        return payload

class TelemetryCompactor:
    """
    Maintains entity state cache and replaces redundant full sensor dumps
    with state-change diffs.
    """
    def __init__(self):
        self._last_states: Dict[str, str] = {}

    def compact_state_dict(self, current_states: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes { 'living_room_temp': '72', 'hvac': 'idle' ... }
        Returns only entries that changed since last invocation.
        """
        diffs = {}
        for entity_id, val in current_states.items():
            str_val = str(val)
            if self._last_states.get(entity_id) != str_val:
                diffs[entity_id] = val
                self._last_states[entity_id] = str_val
        
        if not diffs:
            return {"status": "all_states_unchanged", "entity_count": len(current_states)}
        return diffs
