# Token Watch 🛰️

**Active Token Monitoring and Reduction Advisement System for Multi-Node, Multi-Agent Home Networks.**

Token Watch provides distributed telemetry, real-time token budgeting, prompt slot decomposition, active reduction advisories, and runtime optimizations across edge nodes (RPi, Mini-PCs), local inference runtimes (vLLM, Ollama), and cloud LLM gateways.

---

## 🌟 Key Features

1. **Production-Ready Streaming Sidecar Proxies**:
   - Zero-overhead ASGI / reverse-proxy interceptors per node with full SSE streaming (`stream: true`) passthrough.
   - Offline disk spooling (SQLite WAL buffer) on edge nodes for zero-loss telemetry during network disconnects.
   - Automated response `tool_calls` extraction and provider-native usage ingestion (`cached_tokens`).

2. **Active Optimization & Runtime Interception**:
   - **Dynamic Tool Pruner**: Ranks tools using query-keyword scoring and strips dormant tool schemas prior to upstream inference.
   - **Prefix Cache Stabilizer**: Reorders prompt structures to place static instructions and schemas before dynamic variables, locking in 80%+ prompt caching.
   - **Telemetry Compactor**: Converts repetitive, static IoT state tables into state-change diffs.

3. **Time-Series Analytics & Advisory Engine**:
   - Embedded **DuckDB** storage engine with sub-millisecond query aggregation.
   - **Dormant Tool Pruner Advisor**: Flags unused tool schemas polluting context.
   - **Multi-Agent Ping-Pong Breaker**: Detects recursive inter-agent chatter loops and triggers circuit breakers.
   - **Model Cascading Advisor**: Recommends routing low-complexity agent tasks to faster, cheaper models.

4. **Unified Multi-Model Cost Matrix**:
   - Calculates exact USD cost across Anthropic (Claude 3.5 Sonnet/Haiku/Opus), OpenAI (GPT-4o, o1, o3-mini), Google Gemini (2.5 Flash, Pro), and Local Models (Llama 3, Qwen).
   - Accurately discounts cached input tokens.

5. **Terminal UI & Real-Time Dashboard**:
   - Interactive live console dashboard built with Rich.

---

## 🏗️ Architecture

```
[ Edge Nodes (RPi / ESP / Jetson) ]     [ Local GPU Server (vLLM) ]     [ Orchestrator Node ]
           │                                      │                               │
    (TokenWatch Sidecar)                   (TokenWatch Sidecar)            (TokenWatch Sidecar)
           └──────────────────────────────────────┼───────────────────────────────┘
                                                  │ (Async Telemetry Events)
                                                  ▼
                                      ┌───────────────────────┐
                                      │    TOKEN WATCH HUB    │
                                      │ ───────────────────── │
                                      │ • Slot Decomposer     │
                                      │ • DuckDB Time-Series  │
                                      │ • Circuit Breakers    │
                                      │ • Advisory Engine     │
                                      │ • Rich TUI Dashboard  │
                                      └───────────────────────┘
```

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/twan102000-Builder/token-watch.git
cd token-watch

# Install dependencies (DuckDB, FastAPI, Uvicorn, Rich, Tiktoken)
pip install -e ".[dev]"
```

### 1. Starting the Central Hub

```bash
python -m tokenwatch.hub --port 9000
```

### 2. Running a Node Sidecar

```bash
# Intercepts requests on port 8080 and forwards to upstream target with active tool pruning enabled
python -m tokenwatch.sidecar \
  --port 8080 \
  --upstream http://localhost:11434/v1 \
  --hub-url http://localhost:9000 \
  --node-id node_gpu_livingroom \
  --optimize-tools
```

### 3. Launching the Real-Time Terminal Dashboard

```bash
python -m tokenwatch.cli dashboard --hub-url http://localhost:9000
```

---

## 🧪 Running Tests

```bash
pytest -v tests/test_tokenwatch.py
```

---

## 📄 License

MIT
