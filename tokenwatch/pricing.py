from typing import Dict, Any, Tuple

# Pricing per million tokens (USD)
# (input_per_m, output_per_m, cache_read_per_m, cache_write_per_m)
MODEL_PRICING: Dict[str, Tuple[float, float, float, float]] = {
    # Anthropic
    "claude-3-5-sonnet-20241022": (3.00, 15.00, 0.30, 3.75),
    "claude-3-5-haiku-20241022": (0.80, 4.00, 0.08, 1.00),
    "claude-3-opus-20240229": (15.00, 75.00, 1.50, 18.75),
    
    # OpenAI
    "gpt-4o": (2.50, 10.00, 1.25, 2.50),
    "gpt-4o-mini": (0.15, 0.60, 0.075, 0.15),
    "o1": (15.00, 60.00, 7.50, 15.00),
    "o3-mini": (1.10, 4.40, 0.55, 1.10),
    
    # Google Gemini
    "gemini-1.5-pro": (1.25, 5.00, 0.3125, 1.25),
    "gemini-1.5-flash": (0.075, 0.30, 0.01875, 0.075),
    "gemini-2.0-flash": (0.10, 0.40, 0.025, 0.10),
    "gemini-2.5-flash": (0.10, 0.40, 0.025, 0.10),
    
    # Local models (vLLM / Ollama)
    "local-llama-3.1-8b": (0.00, 0.00, 0.00, 0.00),
    "local-llama-3.3-70b": (0.00, 0.00, 0.00, 0.00),
    "local-qwen-2.5-7b": (0.00, 0.00, 0.00, 0.00),
}

def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_read_tokens: int = 0,
    cached_write_tokens: int = 0
) -> float:
    """
    Calculates cost in USD for a completion call based on model pricing matrix.
    """
    # Normalize model name matching
    matched_key = None
    lower_model = model.lower()
    for key in MODEL_PRICING:
        if key in lower_model:
            matched_key = key
            break
            
    if not matched_key:
        # Default fallback to gpt-4o-mini-style tier if unknown
        inp_rate, out_rate, read_rate, write_rate = (0.50, 2.00, 0.25, 0.50)
    else:
        inp_rate, out_rate, read_rate, write_rate = MODEL_PRICING[matched_key]

    # Non-cached input tokens
    billable_input = max(0, input_tokens - cached_read_tokens)

    cost = (
        (billable_input / 1_000_000.0) * inp_rate
        + (output_tokens / 1_000_000.0) * out_rate
        + (cached_read_tokens / 1_000_000.0) * read_rate
        + (cached_write_tokens / 1_000_000.0) * write_rate
    )
    return round(cost, 6)
