# Smart Notify for Mobility: Geo-Deferred Notification System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![ML](https://img.shields.io/badge/ML-GradientBoosting%20%2B%20IsolationForest-orange.svg)](#ai-and-ml-approach)
[![Data](https://img.shields.io/badge/Data-OpenCelliD%20%2B%20Synthetic-6f42c1.svg)](#data-pipeline)

Smart Notify is an AI-first mobility communication engine for moving users, fleets, and transit corridors. Instead of sending every update immediately, it predicts near-future connectivity, defers non-urgent traffic to better signal windows, and enforces hard safety bounds for urgent and timeout scenarios.

## Mobility Theme Fit

This solution is designed for mobility systems where network quality changes every minute:

- Ride-hailing and taxi ETA updates
- Delivery fleet dispatch and status pushes
- Public transit rider alerts
- Logistics corridor notifications in highway and tunnel segments

The core mobility question we answer is simple: send now, or wait a few seconds for a better radio window without breaking SLA?

## What Problem We Solve

Mobile notifications often arrive in poor network conditions, causing retries, wasted data, delayed user value, and battery drain.

Smart Notify addresses this by:

- Predicting whether signal quality will improve in the next few seconds.
- Deferring non-urgent sends when waiting is likely beneficial.
- Sending urgent notifications immediately.
- Enforcing bounded delay so nothing waits forever.

Mobility-specific value:

- Better message reliability while vehicles are in motion
- Lower retransmission overhead in patchy corridor coverage
- Priority guarantees for critical mobility events

## Why This Is Hackathon-Relevant

This project combines:

- Real-world geospatial context (routes, towers, dead-zones).
- Real-time AI decisioning under uncertainty.
- Practical product tradeoffs between data efficiency and delivery latency.
- End-to-end reproducibility with metrics and artifacts.

## Mobility KPIs We Report

- Timeout send rate
- Urgent send rate
- Data saved versus naive immediate-send baseline
- Per-route data saved (city, highway, tunnel, mixed)
- Decision mix by route and scenario profile

## AI and ML Approach

The decision stack is hybrid and robust:

1. Gradient Boosting Classifier
- Predicts probability of good near-future signal.
- Uses rolling features such as mean and variance of filtered RSSI, slope, delta, speed, and time-since-good.

2. Isolation Forest
- Flags unstable or anomalous signal regimes where sending is likely wasteful.

3. Kalman Filter
- Smooths noisy RSSI observations before model inference.

4. Policy Layer
- Priority-aware thresholds and queue-age logic.
- Urgent and timeout guardrails guarantee delivery progress.

5. Heuristic Fallback
- Keeps the system operational if ML artifacts are unavailable.

## System Architecture

1. Route generation
2. Vehicle trajectory simulation
3. Signal simulation from tower geometry and channel effects
4. Kalman smoothing
5. ML training and validation
6. Real-time inference and decisioning via API and WebSocket
7. SQLite logging and evaluation artifacts

Core modules:

- data/generate_routes.py
- data/simulate_vehicle.py
- data/signal_simulator.py
- signal_processing/kalman_filter.py
- ml/train_models.py
- engine/decision_engine.py
- simulation/realtime_loop.py
- api/main.py
- scripts/run_route_benchmark.py
- scripts/baseline_eval.py

## Data Pipeline

The project supports both realism and offline reliability:

- OpenCelliD tower ingestion when API quota is available.
- Cached OpenCelliD replay mode for deterministic reruns.
- Synthetic fallback behavior for robustness.

Pipeline entrypoint:

- run_pipeline.py

## Benchmark Profiles

Two benchmark profiles are intentionally supported for transparent evaluation:

1. Full profile
- Routes: city, highway, tunnel, mixed
- Notification pressure: higher
- Purpose: stress test under harder network regimes

2. Legacy profile
- Routes: city, tunnel
- Notification pressure: moderate
- Purpose: clean baseline for signal-aware gains

## Latest Measured Results

### Full profile (all scenarios)
Source artifact: artifacts/baseline_summary.md

- Total deliveries: 1090
- Timeout send rate: 28.62%
- Urgent send rate: 20.0%
- Expected data saved: 5.58 KB (0.10%)

Per-route saved metric:

- mixed_route: 10.06 KB (0.33%)
- tunnel_route: 4.13 KB (0.34%)
- city_route: 1.96 KB (0.29%)
- highway_route: -10.57 KB (-1.89%)

### Legacy profile (high-signal baseline)
Source artifact from legacy benchmark run:

- Total deliveries: 419
- Timeout send rate: 3.34%
- Urgent send rate: 20.29%
- Expected data saved: 16.35 KB (1.05%)

Per-route saved metric:

- tunnel_route: strong positive
- city_route: strong positive

## Quick Start

### 1) Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Build data and models

Cache-only OpenCelliD mode (recommended for reproducibility):

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --force --opencellid-cache-only
```

### 3) Run benchmark

Full profile:

```powershell
.\.venv\Scripts\python.exe scripts\run_route_benchmark.py --profile full
.\.venv\Scripts\python.exe scripts\baseline_eval.py
```

Legacy profile:

```powershell
.\.venv\Scripts\python.exe scripts\run_route_benchmark.py --profile legacy
.\.venv\Scripts\python.exe scripts\baseline_eval.py
```

## API and Live Demo

Start API:

```powershell
cd api
..\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
```

Interactive docs:

- http://127.0.0.1:8000/docs

Key endpoints:

- GET /
- GET /routes
- POST /simulate/start
- POST /simulate/stop
- POST /notify
- GET /metrics
- GET /history
- GET /towers
- WS /ws/simulation

## Demo Script for Judges (3-5 minutes)

1. Start API and open docs.
2. Run city route live to show stable behavior.
3. Inject urgent and non-urgent notifications and explain decision differences.
4. Show baseline summary for full profile to prove stress testing across scenarios.
5. Show legacy profile summary to highlight peak optimization under cleaner coverage.
6. Explain transparent tradeoff: robustness versus efficiency in harsh coverage.
7. Close with mobility impact: fewer failed corridor sends and better SLA compliance for moving users.

## What Makes This Project Credible

- End-to-end pipeline from data generation to real-time serving.
- Route-aware, time-aware validation and saved artifacts.
- Reproducible commands and benchmark profiles.
- Explicitly logged decisions and outcomes in SQLite for auditability.

## Known Limits and Next Steps

Current limitation:

- Highway and mixed corridors can enter prolonged low-coverage periods, reducing data-saved performance.

Planned improvements:

- Better corridor-aware tower augmentation.
- Route-conditional policy calibration.
- Online adaptation of thresholds by live coverage confidence.

## Repository Guide

- API: api
- Decision logic: engine
- Data generation: data
- Signal processing: signal_processing
- ML training: ml
- Real-time simulation: simulation
- Scripts and evaluation: scripts
- Benchmark artifacts: artifacts

## Contact and Pitch Summary

Smart Notify demonstrates practical, deployable AI decisioning for mobility networks: less waste, bounded latency, and transparent per-route metrics under both normal and worst-case movement scenarios.
