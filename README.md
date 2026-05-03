# Real-Time AI-Powered Network Intrusion Detection System (NIDS)

A production-grade, distributed intrusion detection system that monitors live network traffic, detects malicious activity using a hybrid ML pipeline, and delivers real-time alerts through a React dashboard and multi-channel notification system.

---

## Architecture

```
Network Traffic (packets)
        │
        ▼
┌───────────────────┐
│  Packet Capture   │  scapy / Zeek  (src/capture/)
└────────┬──────────┘
         │ raw JSON
         ▼
┌───────────────────┐
│  Kafka Ingestion  │  Topic: raw_packets  (src/ingestion/)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Stream Processing │  PySpark Structured Streaming  (src/processing/)
│  (30s windows)   │  Windowing · Aggregation · Flow reconstruction
└────────┬──────────┘
         │ aggregated flow features
         ▼
┌───────────────────────────────────────────────┐
│              Feature Extraction               │
│  flow_features (29)  ·  statistical (17)      │
│  behavioral (15)  ·  total 46-dim vector      │
└────────┬──────────────────────────────────────┘
         │
         ▼
┌───────────────────────────────────────────────┐
│              ML Detection Engine              │
│                                               │
│  ┌──────────────┐  ┌──────────────────────┐  │
│  │  CNN-GRU /   │  │  Isolation Forest +  │  │
│  │  Transformer │  │  VAE Autoencoder     │  │
│  │  (sequence)  │  │  (anomaly)           │  │
│  └──────┬───────┘  └──────────┬───────────┘  │
│         │                     │              │
│  ┌──────▼─────────────────────▼───────────┐  │
│  │     Signature Engine (10 rules)        │  │
│  │     SYN flood · port scan · SSH BF …   │  │
│  └────────────────────────┬───────────────┘  │
└───────────────────────────┼──────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────┐
│              Fusion Layer                     │
│   score = 0.5·ML + 0.3·Anomaly + 0.2·Sig     │
│   → benign / suspicious / malicious           │
└────────┬──────────────────────────────────────┘
         │
         ├──▶  Security Enrichment
         │      Behavioral profiling (Welford)
         │      Adversarial detection
         │      Threat intelligence (AbuseIPDB / VT)
         │
         ├──▶  Adaptive Learning
         │      ADWIN + Page-Hinkley drift detection
         │      River Adaptive Random Forest (online)
         │
         ▼
┌───────────────────┐       ┌─────────────────────┐
│  Alert Manager    │──────▶│  Notifiers           │
│  (deduplicate)    │       │  Slack · Email · HTTP│
└────────┬──────────┘       └─────────────────────┘
         │
         ├──▶  Elasticsearch  (alert + flow storage)
         ├──▶  Kafka: alerts topic
         └──▶  FastAPI + WebSocket ──▶  React Dashboard
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Packet capture | scapy, Zeek |
| Stream ingestion | Apache Kafka (Confluent) |
| Stream processing | PySpark Structured Streaming |
| ML / Deep learning | PyTorch (CNN-GRU, Transformer, VAE) |
| Anomaly detection | scikit-learn Isolation Forest |
| Online learning | River Adaptive Random Forest |
| Drift detection | ADWIN, Page-Hinkley |
| Backend API | FastAPI + WebSocket |
| Storage | Elasticsearch 8, Redis 7 |
| Alerting | Slack SDK, aiosmtplib, httpx |
| Observability | Grafana, Prometheus, Kibana |
| Frontend | React 18 + Recharts + Tailwind CSS |
| Containers | Docker Compose |

---

## Prerequisites

- Docker ≥ 24 and Docker Compose v2
- Python 3.11+ (for local development)
- Node 20+ (for dashboard development)
- Java 17+ (for PySpark — bundled in the processor container)
- `NET_ADMIN` / `NET_RAW` capability on the capture host (for raw packet capture)

---

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your Slack token, SMTP credentials, API keys, etc.

# 2. Start the full stack
docker compose up -d

# 3. Wait for all services to be healthy (~60s)
docker compose ps

# 4. Open the dashboard
open http://localhost:3000

# 5. Simulate traffic to see detections
python scripts/simulate_traffic.py --rate 200 --attack-mix 0.15 --duration 120
```

**Service URLs:**

| Service | URL |
|---|---|
| React Dashboard | http://localhost:3000 |
| NIDS REST API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Kibana | http://localhost:5601 |
| Grafana | http://localhost:3001 (admin / nids_admin) |
| Prometheus | http://localhost:9090 |
| Elasticsearch | http://localhost:9200 |

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and fill in:

```
# Kafka (automatically configured in Docker)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# ML detection weights (must sum to 1.0)
FUSION_WEIGHT_ML=0.50
FUSION_WEIGHT_ANOMALY=0.30
FUSION_WEIGHT_SIGNATURE=0.20

# Alert thresholds
THRESHOLD_SUSPICIOUS=0.40
THRESHOLD_MALICIOUS=0.70

# Slack / SMTP / Webhook (optional — alerts still go to ES if unset)
SLACK_BOT_TOKEN=xoxb-...
SMTP_USER=alerts@yourcompany.com
WEBHOOK_URL=https://hooks.yourcompany.com/nids
```

---

## Running Components Individually

```bash
# Install Python dependencies
pip install -r requirements.txt

# Packet capture (requires root / NET_RAW)
sudo python -m src.capture.packet_capture

# Zeek log tail
python -m src.capture.zeek_parser

# Stream processor (needs running Kafka + ES + Redis)
python -m src.processing.stream_processor

# API server
uvicorn src.api.main:app --reload --port 8000

# React dashboard
cd dashboard && npm install && npm start
```

---

## Training the Models

Download the [CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset and place the CSV files in `./data/`:

```bash
python scripts/train_models.py \
  --dataset cicids \
  --data-dir ./data \
  --model-dir ./models \
  --epochs 30 \
  --batch-size 256

# Skip anomaly model retraining
python scripts/train_models.py --skip-anomaly
```

Trained artifacts are saved to `./models/`:
- `cnn_gru.pt` — CNN-GRU sequence model checkpoint
- `isolation_forest.joblib` — fitted Isolation Forest + scaler
- `autoencoder.pt` — VAE checkpoint + calibrated threshold

---

## Traffic Simulation

The simulation script generates synthetic labelled flows and publishes them to Kafka:

```bash
# 500 flows/s, 10% attack mix, run for 5 minutes
python scripts/simulate_traffic.py \
  --rate 500 \
  --attack-mix 0.10 \
  --duration 300

# Dry-run: print JSON to stdout without publishing
python scripts/simulate_traffic.py --dry-run --attack-mix 0.5
```

Simulated attack profiles: `syn_flood`, `port_scan`, `ssh_brute_force`, `udp_flood`, `slowloris`, `dns_amplification`.

---

## API Reference

### Alerts

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/alerts/` | List recent alerts (filterable) |
| GET | `/api/v1/alerts/{id}` | Get a single alert |
| GET | `/api/v1/alerts/counts` | Alert counts by attack class |
| GET | `/api/v1/alerts/top-sources` | Top attacker IPs |
| POST | `/api/v1/alerts/feedback` | Submit analyst label for online learning |

### Metrics

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/metrics/summary` | Throughput, accuracy, drift status |
| GET | `/api/v1/metrics/traffic` | Traffic volume time series |
| GET | `/api/v1/metrics/anomaly-scores` | Anomaly score histogram |
| GET | `/api/v1/metrics/health` | Redis/API health check |

### Models

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/models/status` | Model load status + online learner stats |
| GET | `/api/v1/models/rules` | List all signature rules |
| DELETE | `/api/v1/models/rules/{id}` | Disable a signature rule |

### WebSocket

`ws://localhost:8000/ws` — pushes `{"type": "alert", "payload": {...}}` frames in real time. Also sends `{"type": "heartbeat"}` every 30 seconds.

---

## Detection Pipeline Details

### Fusion Formula

```
fusion_score = 0.5 × ML_score + 0.3 × anomaly_score + 0.2 × signature_score

threat_label = "malicious"   if fusion_score ≥ 0.70
             = "suspicious"  if fusion_score ≥ 0.40
             = "benign"      otherwise
```

Weights are configurable via `FUSION_WEIGHT_*` environment variables.

### Signature Rules

| ID | Name | Severity |
|---|---|---|
| SIG-001 | SYN Flood | HIGH |
| SIG-002 | Horizontal Port Scan | MEDIUM |
| SIG-003 | UDP Flood | HIGH |
| SIG-004 | ICMP Flood | MEDIUM |
| SIG-005 | SSH Brute Force | HIGH |
| SIG-006 | DNS Amplification | HIGH |
| SIG-007 | TCP Null Scan | MEDIUM |
| SIG-008 | FTP Brute Force | MEDIUM |
| SIG-009 | Slowloris DoS | HIGH |
| SIG-010 | Network Reconnaissance | LOW |

### Adversarial Detection

Detects evasion techniques that deliberately avoid threshold-based rules:
- **Low-and-slow** — artificially low packet rate over long duration
- **Fragmentation abuse** — high MF-flag ratio to scatter payload across fragments
- **Payload mimicry** — entropy-matched payloads designed to look like HTTP/DNS
- **Size padding** — unnaturally uniform packet sizes (fixed-size padding)
- **Protocol tunneling** — data exfiltration via oversized DNS/ICMP payloads

### Adaptive Learning

The `ModelDriftMonitor` runs ADWIN and Page-Hinkley detectors on accuracy, F1, and FPR. When drift is confirmed, the online Adaptive Random Forest retrains incrementally. Analyst feedback submitted via `/api/v1/alerts/feedback` is incorporated immediately without a full retrain.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Individual suites
pytest tests/test_features.py -v
pytest tests/test_detection.py -v
pytest tests/test_api.py -v
```

---

## Project Structure

```
.
├── docker-compose.yml         Full stack (Kafka, ES, Redis, Grafana, Prometheus)
├── Dockerfile.api             FastAPI + WebSocket server
├── Dockerfile.processor       PySpark streaming job
├── Dockerfile.capture         Raw packet capture
├── requirements.txt
├── setup.py
├── .env.example
│
├── src/
│   ├── config.py              Centralised pydantic-settings
│   ├── capture/               Packet capture (scapy, Zeek parser)
│   ├── ingestion/             Kafka producer / consumer
│   ├── processing/            PySpark streaming, windowing
│   ├── features/              Flow, statistical, behavioral extractors
│   ├── detection/
│   │   ├── models/            CNN-GRU, Transformer, IsolationForest, VAE
│   │   ├── signature_engine.py
│   │   └── fusion_layer.py
│   ├── security/              Behavioral profiling, adversarial detection, threat intel
│   ├── adaptive/              ADWIN/Page-Hinkley drift, River online learner
│   ├── alerting/              AlertManager + Slack/Email/Webhook notifiers
│   ├── api/                   FastAPI app, routes, schemas
│   └── storage/               Elasticsearch + Redis clients
│
├── dashboard/                 React + Recharts + Tailwind
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/        LiveAlerts, AlertTimeline, TrafficHeatmap, MetricsPanel
│   │   └── hooks/useWebSocket.ts
│   └── Dockerfile
│
├── scripts/
│   ├── train_models.py        CICIDS/NSL-KDD training pipeline
│   └── simulate_traffic.py    Synthetic traffic + attack generation
│
├── tests/
│   ├── test_features.py
│   ├── test_detection.py
│   └── test_api.py
│
├── config/
│   └── prometheus.yml
│
└── models/                    Saved model checkpoints (git-ignored)
```

---

## Evaluation Metrics

| Category | Metrics |
|---|---|
| Detection | Accuracy, Precision, Recall, F1-score |
| System | Latency (p50/p95), Throughput (pps), Kafka lag |
| Security | False positive rate, Zero-day detection rate |
| Adaptive | Drift events / 24h, Online learner Cohen's Kappa |

---

## Dataset

The training pipeline supports:
- **[CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)** — modern labeled network traffic with DoS, DDoS, brute-force, web attacks, and infiltration scenarios
- **[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)** — classic benchmark for IDS research

---

## Contact

If you come across any issues, have suggestions for improvement, or want to discuss further enhancements, feel free to contact me at [jaya2004kra@gmail.com](mailto:jaya2004kra@gmail.com). Your feedback is greatly appreciated.

---

## License

All the code and resources in this repository are licensed under the GNU General Public License. You are free to use, modify, and distribute the code under the terms of this license. However, I do not take responsibility for the accuracy or reliability of the programs.

