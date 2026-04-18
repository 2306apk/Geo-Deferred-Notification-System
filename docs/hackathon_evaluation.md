# Smart Notify Hackathon Evaluation and Demo Narrative

## What We Are Solving
A moving-vehicle notification engine should avoid sending in poor/unstable signal windows, defer when better connectivity is likely soon, and still enforce bounded delay through timeout delivery.

## Evaluation Protocol
1. Run route scenarios for city, highway, and tunnel.
2. Collect API metrics and delivery history after each run.
3. Generate baseline report from SQLite logs.
4. Compare decision mix, send-time signal quality, and timeout behavior across routes.

## Recommended Commands (Windows PowerShell)

From repo root:

1. Start API
- cd api
- ..\\.venv\\Scripts\\python.exe -m uvicorn main:app --reload --port 8000

In a second terminal:

2. Start a 60-second run
- $base = "http://127.0.0.1:8000"
- Invoke-RestMethod -Method Post "$base/simulate/start" -ContentType "application/json" -Body (@{route="city_route";speed_factor=20;notif_rate=6} | ConvertTo-Json)
- Start-Sleep -Seconds 60
- Invoke-RestMethod -Method Post "$base/simulate/stop"

3. Build baseline summary
- python scripts/baseline_eval.py

## Metrics to Present to Judges
- Decision mix percentages: SEND_NOW, WAIT_BETTER, WAIT_ANOMALY, SEND_TIMEOUT, SEND_URGENT
- Average wait time before delivery
- Average RSSI/quality at send time
- Timeout rate (shows bounded-delay behavior)
- Per-route breakdown (city/highway/tunnel)

## How to Narrate Results
1. Explain that the model predicts short-horizon future connectivity, not just current signal.
2. Show that tunnel/unstable environments create meaningful defer decisions.
3. Show that urgent and timeout safeguards prevent indefinite waiting.
4. Highlight improved send-time quality and controlled delay as the key system tradeoff.

## Integrity Notes
- Validation includes route-holdout and time-ordered splits to reduce optimistic leakage.
- Minority-class behavior is reported explicitly (class-0 recall/precision, balanced accuracy, PR-AUC for both classes).
- If ML models are unavailable, the decision engine falls back to deterministic heuristics and remains operational.
