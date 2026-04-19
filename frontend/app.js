const API_BASE = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws/simulation";
const DATASET_URL = "./data/dataset.csv";
const APP_CONFIG = window.__APP_CONFIG__ || {};

const state = {
  route: [],
  routeName: "city_route",
  currentIndex: 0,
  currentFrame: null,
  pending: [],
  delivered: [],
  lastPrediction: null,
  map: null,
  carMarker: null,
  deliveredLayer: null,
  baseLayer: null,
  mapStyle: "dark",
  stableGoodSteps: 0,
  recentDropSteps: 999,
  simulationRunning: false,
};

const STEP_SECONDS = 5;
const STABLE_GOOD_STEPS_REQUIRED = 3;
const HANDOFF_GRACE_STEPS = 1;
const DELIVERY_BATCH_LIMIT = 3;
const MAX_QUEUE_SIZE = 8;
const QUEUE_TIMEOUT_STEPS = 18;

const TILE_SOURCES = {
  satellite: {
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    options: { maxZoom: 18 },
    label: "Fallback satellite",
  },
  dark: {
    url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    options: { maxZoom: 19, subdomains: "abcd" },
    label: "Carto styled dark",
  },
};

function timestampNow() {
  return new Date().toISOString().slice(0, 19).replace("T", " ");
}

function kmhToMph(kmh) {
  return kmh * 0.621371;
}

function collapseDeferredQueue() {
  if (state.pending.length <= MAX_QUEUE_SIZE) return;

  const urgent = state.pending.filter((item) => item.priority === "urgent");
  const deferred = state.pending.filter((item) => item.priority !== "urgent");

  if (urgent.length >= MAX_QUEUE_SIZE) {
    state.pending = urgent.slice(0, MAX_QUEUE_SIZE);
    return;
  }

  const roomForDeferred = Math.max(0, MAX_QUEUE_SIZE - urgent.length - 1);
  const keptDeferred = deferred.slice(0, roomForDeferred);
  const collapsedCount = deferred.length - keptDeferred.length;

  if (collapsedCount > 0) {
    keptDeferred.push({
      id: `summary-${Date.now()}`,
      text: `${collapsedCount} deferred notifications summarized`,
      base_text: "Deferred queue summary",
      priority: "deferred",
      created_at: timestampNow(),
      created_index: state.currentIndex,
      repeat_count: collapsedCount,
      is_summary: true,
    });
  }

  state.pending = [...urgent, ...keptDeferred];
}

function parseCsv(text) {
  const [headerLine, ...lines] = text.trim().split("\n");
  const headers = headerLine.split(",");
  return lines.map((line) => {
    const parts = line.split(",");
    const row = {};
    headers.forEach((header, idx) => {
      const raw = parts[idx];
      const numeric = Number(raw);
      row[header] = Number.isNaN(numeric) ? raw : numeric;
    });
    return row;
  });
}

function currentRow() {
  return state.currentFrame || state.route[state.currentIndex] || state.route[0] || {};
}

function currentPayload() {
  const row = currentRow();
  return {
    speed: row.speed || row.speed_kmh || 0,
    accel: row.accel,
    signal: row.signal,
    signal_1: row.signal_1,
    signal_2: row.signal_2,
    time_since_last_good: row.time_since_last_good,
    signal_bars: row.signal_bars,
    rsrp_dbm: row.rsrp_dbm,
    rsrq_db: row.rsrq_db,
    sinr_db: row.sinr_db,
    tower_distance_m: row.tower_distance_m,
    tower_count_nearby: row.tower_count_nearby,
    real_tower_visible: row.real_tower_visible,
    rsrp_1: row.rsrp_1,
    sinr_1: row.sinr_1,
    signal_delta: row.signal_delta,
    rsrp_delta: row.rsrp_delta,
    handover: row.handover,
    distance_since_last_good_m: row.distance_since_last_good_m,
    stable_good_seconds: state.stableGoodSteps * STEP_SECONDS,
    recent_drop_seconds: state.recentDropSteps * STEP_SECONDS,
    gps_accuracy_m: 12,
    queue_size: state.pending.length,
    active_network: "cellular",
  };
}

function displaySpeedMph(row) {
  const kmh = Number(row.speed_kmh);
  if (Number.isFinite(kmh)) return kmhToMph(kmh);
  const ms = Number(row.speed);
  return Number.isFinite(ms) ? kmhToMph(ms * 3.6) : 0;
}
function injectDemoNotifications() {
  const messages = [
    "📦 Delivery update",
    "⚠️ Traffic ahead",
    "📍 New pickup assigned",
    "🚗 Route deviation alert",
    "🔔 Reminder: Check order",
  ];

  const msg = messages[Math.floor(Math.random() * messages.length)];

  const notif = {
    id: `demo-${Date.now()}`,
    text: msg,
    base_text: msg,
    priority: Math.random() > 0.7 ? "urgent" : "deferred",
    created_at: timestampNow(),
    created_index: state.currentIndex,
  };

  state.pending.push(notif);
  collapseDeferredQueue();
}

async function predict(payload) {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data: payload }),
  });
  if (!response.ok) {
    throw new Error(`Prediction request failed: ${response.status}`);
  }
  return response.json();
}

function decisionClass(decision) {
  if (decision.startsWith("SEND")) return "good";
  if (decision.startsWith("WAIT")) return "warn";
  return "bad";
}

function premiumSatelliteSource() {
  if (APP_CONFIG.maptilerKey) {
    return {
      url: `https://api.maptiler.com/tiles/satellite-v4/{z}/{x}/{y}.jpg?key=${APP_CONFIG.maptilerKey}`,
      options: { maxZoom: 20, tileSize: 512, zoomOffset: -1 },
      label: "MapTiler satellite",
    };
  }
  if (APP_CONFIG.mapboxToken) {
    return {
      url: `https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/tiles/512/{z}/{x}/{y}@2x?access_token=${APP_CONFIG.mapboxToken}`,
      options: { maxZoom: 20, tileSize: 512, zoomOffset: -1 },
      label: "Mapbox satellite streets",
    };
  }
  return { ...TILE_SOURCES.satellite, label: "Fallback satellite" };
}
function safeNum(val, digits = 2) {
  const n = Number(val);
  return Number.isFinite(n) ? n.toFixed(digits) : "-";
}

async function evaluateCurrentState() {
  if (state.currentFrame && state.currentFrame.prediction) {
    state.lastPrediction = state.currentFrame.prediction;
    renderStatus();
    return;
  }
  try {
    state.lastPrediction = await predict(currentPayload());
  } catch (err) {
    console.warn("evaluateCurrentState: predict failed", err);
    state.lastPrediction = null;
  }
  renderStatus();
}

function renderStatus() {
  const row = currentRow() || {};
  const coverageGood = Number(row.signal) === 1;
  const prediction = state.lastPrediction;

  document.getElementById("statusCoords").textContent = `${safeNum(row.lat, 3)}, ${safeNum(row.lon, 3)}`;
  document.getElementById("statusSpeed").textContent = `${safeNum(displaySpeedMph(row), 1)} mph`;
  document.getElementById("statusCoverage").textContent = coverageGood ? "Good coverage" : "Poor coverage";
  document.getElementById("statusPending").textContent = `${state.pending.length} pending`;
  document.getElementById("statusDelivered").textContent = `${state.delivered.length} delivered`;

  document.getElementById("metricCoverage").textContent = coverageGood ? "Good" : "Poor";
  document.getElementById("metricCoverage").style.color = coverageGood ? "var(--good)" : "var(--bad)";
  document.getElementById("metricRsrp").textContent = `RSRP ${safeNum(row.rsrp_dbm, 1)} dBm`;
  document.getElementById("metricPending").textContent = `${state.pending.length}`;
  document.getElementById("metricDelivered").textContent = `Delivered ${state.delivered.length}`;
  document.getElementById("metricSpeed").textContent = `${safeNum(displaySpeedMph(row), 1)} mph`;
  document.getElementById("metricSinr").textContent = `SINR ${safeNum(row.sinr_db, 1)} dB`;

  if (prediction) {
    document.getElementById("metricConfidence").textContent = safeNum(prediction.confidence, 2);
    document.getElementById("metricDecision").textContent = `${prediction.decision || "-"} · ${prediction.model_used || "-"}`;
    const prob = (prediction.prob_good_signal ?? prediction.probability);
    document.getElementById("metricProbability").textContent = (typeof prob !== "undefined" && prob !== null) ? `P(send) ${safeNum(prob, 2)}` : "-";

    if (typeof prediction.distraction_risk !== "undefined") {
      const dr = Number(prediction.distraction_risk) || 0;
      document.getElementById("metricDistraction").textContent = safeNum(dr, 2);
      const el = document.getElementById("metricDistraction");
      if (dr <= 0) el.style.color = "var(--good)";
      else if (dr < 2) el.style.color = "var(--warn)";
      else el.style.color = "var(--bad)";
    }

    const pill = document.getElementById("statusDecision");
    const suffix = prediction.edge_case_applied ? " · guardrail" : " · model";
    pill.textContent = `${prediction.decision || "-"} · confidence ${safeNum(prediction.confidence, 2)}${suffix}`;
    pill.style.color = `var(--${decisionClass(prediction.decision || "")})`;
    pill.style.background = (prediction.decision || "").startsWith("SEND")
      ? "rgba(142, 242, 194, 0.08)"
      : (prediction.decision || "").startsWith("WAIT")
        ? "rgba(243, 211, 109, 0.08)"
        : "rgba(255, 149, 135, 0.08)";
    if (prediction.minimal_ui) {
      pill.textContent += " · minimal UI";
      pill.style.opacity = 0.9;
    }
  } else {
    document.getElementById("metricConfidence").textContent = "-";
    document.getElementById("metricDecision").textContent = "-";
    document.getElementById("metricProbability").textContent = "-";
  }
}

function notificationCard(notif, delivered = false) {
  const color = notif.priority === "urgent" ? "🔴" : "🟡";
  const meta = delivered ? `Delivered: ${notif.delivered_at || "-"}` : `Queued: ${notif.created_at || "-"}`;
  const minimalBadge = (state.lastPrediction && state.lastPrediction.minimal_ui) || notif.minimal_ui ? ' · minimal UI' : '';
  const probText = (delivered && Number.isFinite(Number(notif.probability))) ? `P(send): ${safeNum(notif.probability, 2)} · thr: ${notif.decision_threshold ?? '-'}` : '';
  const decisionText = delivered ? `<div class="notif-meta">${notif.decision || '-'} · ${notif.reason || ''}</div>` : '';
  return `
    <div class="notif-card glass">
      <div class="notif-title">${color} ${notif.text || '-'}</div>
      <div class="notif-meta">${meta}${minimalBadge}</div>
      ${probText ? `<div class="notif-meta">${probText}</div>` : ''}
      ${decisionText}
    </div>
  `;
}

function renderNotifications() {
  const pendingList = document.getElementById("pendingList");
  const deliveredList = document.getElementById("deliveredList");

  pendingList.innerHTML = state.pending.length
    ? state.pending.map((notif) => notificationCard(notif)).join("")
    : '<div class="notif-empty glass">No pending notifications</div>';

  deliveredList.innerHTML = state.delivered.length
    ? [...state.delivered].reverse().map((notif) => notificationCard(notif, true)).join("")
    : '<div class="notif-empty glass">No deliveries yet</div>';
}

function initMap() {
  const first = state.route[0];
  state.map = L.map("map", {
    zoomControl: true,
    preferCanvas: true,
    attributionControl: false,
  }).setView([first.lat, first.lon], 12);
  setBaseLayer(state.mapStyle);

  const bounds = [];
  for (let i = 0; i < state.route.length - 1; i += 1) {
    const a = state.route[i];
    const b = state.route[i + 1];
    bounds.push([a.lat, a.lon]);
    L.polyline(
      [
        [a.lat, a.lon],
        [b.lat, b.lon],
      ],
      {
        color: Number(a.signal) === 1 ? "#8ef2c2" : "#ff9587",
        weight: 5,
        opacity: 0.92,
      },
    ).addTo(state.map);
  }
  bounds.push([state.route[state.route.length - 1].lat, state.route[state.route.length - 1].lon]);
  state.map.fitBounds(bounds, { padding: [28, 28] });

  const carIcon = L.divIcon({
    className: "",
    html: '<div class="car-marker">🚗</div>',
    iconSize: [46, 46],
    iconAnchor: [23, 23],
  });

  state.carMarker = L.marker([first.lat, first.lon], { icon: carIcon }).addTo(state.map);
  state.deliveredLayer = L.layerGroup().addTo(state.map);
  state.liveTrace = L.polyline([], { color: '#8ef2c2', weight: 4, opacity: 0.9 }).addTo(state.map);
  requestAnimationFrame(() => {
    state.map.invalidateSize();
    state.map.fitBounds(bounds, { padding: [28, 28] });
  });
}
let socket;



function startWsSimulation() {
  console.log("🔌 Connecting WebSocket...");

  // 🔥 close old connection if exists
  if (state._ws && state._ws.readyState === WebSocket.OPEN) {
    state._ws.close();
  }

  const wsUrl = "ws://127.0.0.1:8000/ws/simulation";
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("✅ WS connected");
  };

  ws.onmessage = (event) => {
    try {
      const frame = JSON.parse(event.data);
      console.log("📡 FRAME:", frame);

      // 🔥 update state
      state.currentFrame = frame;
      state.currentIndex = frame.t || 0;
      state.lastPrediction = frame.prediction || null;

      const lat = Number(frame.lat);
      const lon = Number(frame.lon);

      if (!lat || !lon) return;

      // 🔥 move car
      if (state.carMarker) {
        state.carMarker.setLatLng([lat, lon]);
      }

      // 🔥 update UI
      renderStatus();
      renderNotifications();

    } catch (err) {
      console.error("WS error parsing frame:", err);
    }
  };

  ws.onerror = (err) => {
    console.error("❌ WS error:", err);
  };

  ws.onclose = () => {
    console.warn("⚠️ WS closed");
  };

  state._ws = ws;
}

function setBaseLayer(style) {
  state.mapStyle = style;
  if (state.baseLayer) {
    state.map.removeLayer(state.baseLayer);
    state.baseLayer = null;
  }

  if (style === "schematic") {
    document.getElementById("mapProviderHint").textContent = "Provider: schematic local canvas";
    return;
  }

  const source = style === "satellite" ? premiumSatelliteSource() : (TILE_SOURCES[style] || TILE_SOURCES.dark);
  state.baseLayer = L.tileLayer(source.url, source.options).addTo(state.map);
  document.getElementById("mapProviderHint").textContent = `Provider: ${source.label || style}`;
}

function updateMap() {
  const row = currentRow();
  if (!row || !state.carMarker) return;
  state.carMarker.setLatLng([row.lat, row.lon]);
  state.map.panTo([row.lat, row.lon], { animate: true, duration: 0.6 });

  state.deliveredLayer.clearLayers();
  for (const notif of state.delivered) {
    const markerIcon = L.divIcon({
      className: "",
      html: '<div class="delivery-marker"></div>',
      iconSize: [14, 14],
      iconAnchor: [7, 7],
    });
    L.marker([notif.delivery_lat, notif.delivery_lon], { icon: markerIcon })
      .bindPopup(`${notif.text}<br>${notif.delivered_at}`)
      .addTo(state.deliveredLayer);
  }
}

// evaluateCurrentState already defined above (safe wrapper)

function updateEdgeState() {
  const row = currentRow();
  const signalGood = Number(row.signal) === 1;
  const handover = Number(row.handover) >= 1;

  if (signalGood && !handover) {
    state.stableGoodSteps += 1;
    state.recentDropSteps += 1;
  } else if (handover) {
    state.stableGoodSteps = 0;
    state.recentDropSteps = 0;
  } else {
    state.stableGoodSteps = 0;
    state.recentDropSteps = 0;
  }
}

let isRunning = false;

async function advanceCar() {
  console.log("🚗 Advance clicked");

  if (isRunning) {
    console.log("⚠️ Already running");
    return;
  }

  isRunning = true;
  startLocalSimulation();
}

let localInterval = null;

function startLocalSimulation() {
  console.log("🚗 Local simulation started");

  if (localInterval) {
    clearInterval(localInterval);
  }

  state.currentIndex = 0;

  localInterval = setInterval(async () => {
    if (state.currentIndex >= state.route.length) {
      clearInterval(localInterval);
      console.log("✅ Route finished");
      return;
    }

    const row = state.route[state.currentIndex];
    state.currentFrame = row;

    // 🚗 move car
    if (state.carMarker) {
      state.carMarker.setLatLng([row.lat, row.lon]);
    }
    // 🔥 every few steps generate notification
    if (state.currentIndex % 5 === 0) {
      injectDemoNotifications();
    }

    // 🔥 CRITICAL: run prediction
    await evaluateCurrentState();

    // ✅ DELIVERY LOGIC (correct place)
    const decision = state.lastPrediction?.decision || "";

// ✅ DEMO OVERRIDE LOGIC
const shouldDeliver =
  decision.startsWith("SEND") ||
  (state.currentIndex % 10 === 0); // 🔥 force every 10 steps

console.log("Decision:", decision, "| Deliver:", shouldDeliver);

if (shouldDeliver && state.pending.length > 0) {

  const toDeliver = state.pending.splice(0, 1)[0];

  toDeliver.delivered_at = timestampNow();
  toDeliver.delivery_lat = state.currentFrame.lat;
  toDeliver.delivery_lon = state.currentFrame.lon;

  state.delivered.push(toDeliver);

  console.log("📦 Delivered:", toDeliver.text);
}
    state.lastPrediction = state.lastPrediction || {
  confidence: 0.5,
  decision: "WAIT",
  probability: 0.5
};

    // 🔥 update UI
    renderStatus();
    renderNotifications();

    state.currentIndex++;

  }, 500);
  if (state.currentIndex >= state.route.length) {
  clearInterval(localInterval);
  console.log("✅ Route finished");

  isRunning = false; // 🔥 IMPORTANT
  return;
}
}



async function resetApp() {
  try {
    await fetch(`${API_BASE}/simulate/stop`, { method: "POST" });
  } catch (err) {
    console.warn("Reset stop failed", err);
  }
  state.simulationRunning = false;
  state.currentIndex = 0;
  state.currentFrame = null;
  state.pending = [];
  state.delivered = [];
  state.stableGoodSteps = 0;
  state.recentDropSteps = 999;
  if (state.liveTrace) {
    state.liveTrace.setLatLngs([]);
  }
  if (state.route[0] && state.carMarker) {
    state.carMarker.setLatLng([state.route[0].lat, state.route[0].lon]);
  }
  renderNotifications();
  updateMap();
  await evaluateCurrentState();

  await startSimulation(true);
}

async function queueNotification() {
  const text = document.getElementById("messageInput").value.trim();
  const priority = document.getElementById("prioritySelect").value;
  if (!text) return;
  const priorityValue = priority === "urgent" ? 1 : 5;
  try {
    const response = await fetch(`${API_BASE}/notify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, priority: priorityValue }),
    });
    if (!response.ok) {
      throw new Error(`Notify request failed: ${response.status}`);
    }
    const payload = await response.json();
    if (payload.notif) {
      const notif = payload.notif;
      const existing = state.pending.find((item) => item.id === notif.id);
      if (!existing) {
        state.pending.push({
          id: notif.id,
          text: notif.message,
          base_text: notif.message,
          priority: notif.urgent ? "urgent" : "deferred",
          created_at: notif.created_at,
          repeat_count: notif.repeat_count || 1,
        });
      }
    }
  } catch (err) {
    console.error(err);
    return;
  }
  collapseDeferredQueue();
  renderNotifications();
  renderStatus();
}

async function startSimulation(forceRestart = false) {
  if (forceRestart) {
    try {
      await fetch(`${API_BASE}/simulate/stop`, { method: "POST" });
    } catch (err) {
      console.warn("Force restart stop failed", err);
    }
    state.simulationRunning = false;
  }
  if (state.simulationRunning) return;

  const response = await fetch(`${API_BASE}/simulate/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      route: state.routeName,
      speed_factor: 20.0,
      notif_rate: 4.0,
    }),
  });
  if (!response.ok) {
    throw new Error(`Simulation start failed: ${response.status}`);
  }
  state.simulationRunning = true;
}

async function init() {
  const csvText = await fetch(DATASET_URL).then((res) => res.text());
  const allRows = parseCsv(csvText);
  state.route = allRows.filter((row) => row.route === state.routeName);
  if (!state.route.length) {
    state.route = allRows;
  }
  state.pending = [];

  initMap();
  startWsSimulation();
  renderNotifications();
  await evaluateCurrentState();
  await startSimulation();

  document.getElementById("advanceBtn").addEventListener("click", advanceCar);
  document.getElementById("resetBtn").addEventListener("click", resetApp);
  document.getElementById("queueBtn").addEventListener("click", queueNotification);
  document.getElementById("mapStyleSelect").addEventListener("change", (event) => {
    setBaseLayer(event.target.value);
  });
}

init().catch((error) => {
  console.error(error);
  alert(`Failed to initialize frontend: ${error.message}`);
});
