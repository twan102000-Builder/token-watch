# Token Watch 🛰️

**Active Token Monitoring and Reduction Advisement System for Multi-Node, Multi-Agent Home Networks.**

Token Watch provides distributed telemetry, real-time token budgeting, prompt slot decomposition, and active reduction advisories across edge nodes, local inference servers (Ollama, vLLM), and cloud LLM gateways.

---

## 🌟 Key Features

1. **Distributed Sidecar Proxies**:
   - Zero-overhead ASGI / reverse-proxy interceptors per node (RPi, Mini-PCs, GPU rigs).
   - Ingests streaming usage metrics with node, agent, and task attribution headers.

2. **Prompt Slot Decomposition**:
   - Deconstructs LLM payload tokens into discrete semantic components:
     - `system_instructions`
     - `tool_schemas`
     - `iot_telemetry`
     - `conversation_history`
     - `task_payload`

3. **Active Advisement Engine**:
   - **Cache Alignment Auditor**: Identifies prefix jitter to unlock 80%+ prompt caching.
   - **Dormant Tool Pruner**: Highlights unused tool definitions polluting context.
   - **Telemetry Delta Compactor**: Detects repetitive sensor state dumps and suggests diff-based delta ingestion.
   - **Multi-Agent Ping-Pong Breaker**: Real-time loop detection and circuit breaker.

4. **Multi-Model Cost & Quota Management**:
   - Per-agent, per-node sliding-window token budgets.
   - Unified pricing matrix for Cloud (Anthropic, Gemini, OpenAI) and Local models.

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
                                      │ • Circuit Breakers    │
                                      │ • Advisory Engine     │
                                      │ • Storage & Dashboard │
                                      └───────────────────────┘
```

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/twan102000-Builder/token-watch.git
cd token-watch

# Install dependencies
pip install -e .
```

### Starting the Hub

```bash
python -m tokenwatch.hub
```

### Running a Node Sidecar

```bash
# Intercepts requests on port 8080 and forwards to upstream target
python -m tokenwatch.sidecar --port 8080 --upstream http://localhost:11434/v1 --node-id node_gpu
```

---

## 📄 License

MIT
