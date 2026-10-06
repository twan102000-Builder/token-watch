# Token Watch 🛰️

**Active Token Monitoring, Reduction Advisement, and Autonomous Optimization System for Multi-Node, Multi-Agent Home Networks.**

Token Watch is designed to be **100% Free, Private, Offline-Capable, and Self-Contained**. It uses zero external database binaries (powered entirely by Python's built-in `sqlite3` in WAL mode) and transmits zero data to third-party telemetry vendors.

---

## 🌟 Core Highlights

### 🔒 100% Private & LAN Hardened
* **Zero SaaS / Cloud Dependencies**: No LangSmith, Helicone, Datadog, or external telemetry services.
* **Pure Built-In Storage**: Backed exclusively by Python's built-in `sqlite3` with Write-Ahead Logging (`WAL`), running seamlessly on low-power Raspberry Pis and local servers with zero external database compilation.
* **LAN Pre-Shared Key (PSK) Auth**: Enforces `X-TokenWatch-Key` authentication to shield endpoints against untrusted LAN IoT devices.
* **Automated Data Redaction**: Automatically scrubs API keys (`sk-...`, `AIza...`), passwords, Wi-Fi credentials, and alarm PINs before any telemetry or metadata is written to disk.

### ⚡ Active Optimization & Interception
* **Dynamic Tool Pruner**: Ranks tools using query-keyword scoring and strips dormant tool schemas prior to upstream inference (reducing prompt tokens by up to 60%).
* **Prefix Cache Stabilizer**: Reorders prompt structures to place static instructions and schemas before dynamic variables, locking in 80%+ prompt caching.
* **Telemetry Compactor**: Converts repetitive, static IoT state tables into state-change diffs.
* **Active Circuit Breaker**: Detects recursive agent chatter loops (`LOOP_DETECTED`) and blocks runaways with HTTP 429 terminal responses.

### 🌐 Multi-Node Topology & Self-Healing
* **Streaming Sidecar Proxies**: Zero-overhead ASGI / reverse-proxy interceptors per node with full SSE streaming (`stream: true`) passthrough.
* **Offline Spooling Buffer**: Local SQLite buffer caches telemetry on edge nodes during network blips and drains asynchronously to the Hub when connectivity resumes.
* **Heartbeat & Health Registry**: Monitors node health, memory usage, and online/offline status across your home network.
* **Local Home Assistant Alerts**: Dispatches webhook notifications directly to local Home Assistant instances upon circuit trips or budget exhaustion.

---

## 🏗️ Architecture

```
[ Edge Nodes (RPi / ESP / Jetson) ]     [ Local GPU Server (vLLM) ]     [ Orchestrator Node ]
           │                                      │                               │
    (TokenWatch Sidecar)                   (TokenWatch Sidecar)            (TokenWatch Sidecar)
    • Offline SQLite Spool                 • Offline SQLite Spool          • Offline SQLite Spool
    • Secret Redaction                     • Secret Redaction              • Secret Redaction
           └──────────────────────────────────────┼───────────────────────────────┘
                                                  │ (Authenticated Event Stream)
                                                  ▼
                                      ┌───────────────────────┐
                                      │    TOKEN WATCH HUB    │
                                      │ ───────────────────── │
                                      │ • Built-in SQLite WAL │
                                      │ • Circuit Breakers    │
                                      │ • Advisory Engine     │
                                      │ • Node Topology Reg.  │
                                      │ • Rich TUI Console    │
                                      │ • Home Assistant Hook │
                                      └───────────────────────┘
```

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/twan102000-Builder/token-watch.git
cd token-watch

# Install dependencies (only FastAPI, Uvicorn, httpx, pydantic, rich)
pip install -e .
```

### 1. Configuration (`.env`)

Copy the example configuration:
```bash
cp .env.example .env
```
Set your private local key:
```ini
TOKENWATCH_API_KEY=your_secure_local_pre_shared_key
```

### 2. Starting the Central Hub

```bash
python -m tokenwatch.hub --host 127.0.0.1 --port 9000
```

### 3. Running a Node Sidecar

```bash
# Intercepts requests on port 8080 and forwards to upstream target with active tool pruning enabled
python -m tokenwatch.sidecar \
  --port 8080 \
  --upstream http://localhost:11434/v1 \
  --hub-url http://localhost:9000 \
  --node-id node_gpu_livingroom \
  --optimize-tools
```

### 4. Launching the Real-Time Terminal Dashboard

```bash
python -m tokenwatch.cli dashboard --hub-url http://localhost:9000
```

---

## 🛠️ Turnkey Deployment

Pre-configured systemd unit files and Docker Compose are provided in [`deploy/`](file:///Users/anthonymachomefolder/Token%20Watch/deploy/):

* **Systemd Hub**: `deploy/systemd/tokenwatch-hub.service`
* **Systemd Sidecar**: `deploy/systemd/tokenwatch-sidecar.service`
* **Docker Compose**: `deploy/docker-compose.yml`

---

## 🧪 Automated Tests

```bash
pytest -v tests/test_tokenwatch.py
```

---

## 📄 License

MIT
