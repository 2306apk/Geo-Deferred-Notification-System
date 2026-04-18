const API_BASE = "http://127.0.0.1:8000";
const DATASET_URL = "/data/dataset.csv";
const APP_CONFIG = window.__APP_CONFIG__ || {};

const state = {
  route: [],
  currentIndex: 0,
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

function bootstrapNotifications() {
  return [
    { id: "n1", text: "Bank transaction alert: $50 withdrawn", base_text: "Bank transaction alert: $50 withdrawn", priority: "urgent", created_at: timestampNow(), created_index: 0, repeat_count: 1 },
    { id: "n2", text: "Emergency weather alert: Flash flood warning", base_text: "Emergency weather alert: Flash flood warning", priority: "urgent", created_at: timestampNow(), created_index: 0, repeat_count: 1 },
    { id: "n3", text: "Your coffee is ready at Starbucks", base_text: "Your coffee is ready at Starbucks", priority: "deferred", created_at: timestampNow(), created_index: 0, repeat_count: 1 },
    { id: "n4", text: "New promotion: 20% off your next purchase", base_text: "New promotion: 20% off your next purchase", priority: "deferred", created_at: timestampNow(), created_index: 0, repeat_count: 1 },
  ];
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
  return state.route[state.currentIndex];
}

function currentPayload() {
  const row = currentRow();
  return {
    speed: row.speed,
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
  document.getElementById("statusSpeed").textContent = `${safeNum(kmhToMph(row.speed), 1)} mph`;
  document.getElementById("statusCoverage").textContent = coverageGood ? "Good coverage" : "Poor coverage";
  document.getElementById("statusPending").textContent = `${state.pending.length} pending`;
  document.getElementById("statusDelivered").textContent = `${state.delivered.length} delivered`;

  document.getElementById("metricCoverage").textContent = coverageGood ? "Good" : "Poor";
  document.getElementById("metricCoverage").style.color = coverageGood ? "var(--good)" : "var(--bad)";
  document.getElementById("metricRsrp").textContent = `RSRP ${safeNum(row.rsrp_dbm, 1)} dBm`;
  document.getElementById("metricPending").textContent = `${state.pending.length}`;
  document.getElementById("metricDelivered").textContent = `Delivered ${state.delivered.length}`;
  document.getElementById("metricSpeed").textContent = `${safeNum(kmhToMph(row.speed), 1)} mph`;
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

function startWsSimulation() {
  try {
    const wsUrl = API_BASE.replace(/^http/, 'ws') + '/ws/simulation';
    const ws = new WebSocket(wsUrl);
    ws.addEventListener('open', () => console.log('WS connected'));
    ws.addEventListener('message', (evt) => {
      try {
        const frame = JSON.parse(evt.data);
        if (!frame) return;
        const lat = Number(frame.lat || 0);
        const lon = Number(frame.lon || 0);
        if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
        // move car marker
        if (state.carMarker) state.carMarker.setLatLng([lat, lon]);
        // append to live trace
        if (state.liveTrace) {
          const latlngs = state.liveTrace.getLatLngs();
          latlngs.push([lat, lon]);
          state.liveTrace.setLatLngs(latlngs);
        }
        // append any delivered events to UI
        if (Array.isArray(frame.events) && frame.events.length) {
          for (const e of frame.events) {
            const notif = {
              id: e.notif_id || `ws-${Date.now()}`,
              text: e.reason || e.notif_id || 'Delivered',
              priority: 'urgent',
              delivered_at: new Date().toISOString().slice(0,19).replace('T',' '),
              delivery_lat: lat,
              delivery_lon: lon,
              decision: e.decision,
                  reason: e.reason,
                  probability: e.probability ?? e.prob_good_signal ?? null,
                  decision_threshold: e.decision_threshold ?? null,
            };
            state.delivered.push(notif);
          }
          renderNotifications();
        }
      } catch (err) {
        console.error('WS frame error', err);
      }
    });
    ws.addEventListener('close', () => console.log('WS closed'));
    state._ws = ws;
  } catch (e) {
    console.warn('WS connect failed', e);
  }
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

async function advanceCar() {
  if (state.currentIndex >= state.route.length - 1) return;
  state.currentIndex += 1;
  const row = currentRow();
  updateEdgeState();

  // Send current environment to backend /advance which evaluates pending notifications
  const payload = {
    lat: row.lat,
    lon: row.lon,
    speed: row.speed,
    accel: row.accel,
    signal: row.signal,
    rsrp_dbm: row.rsrp_dbm,
    sinr_db: row.sinr_db,
    handover: row.handover,
    gps_accuracy_m: 12,
  };

  try {
    const res = await fetch(`${API_BASE}/advance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`Advance API failed: ${res.status}`);
    const data = await res.json();

    // integrate delivered items returned by backend
    const deliveredFromServer = data.delivered || [];
    for (const d of deliveredFromServer) {
      state.delivered.push({
        ...d,
        delivery_lat: d.delivery_lat ?? row.lat,
        delivery_lon: d.delivery_lon ?? row.lon,
        probability: d.probability ?? d.prob_good_signal ?? null,
        decision_threshold: d.decision_threshold_used ?? d.decision_threshold ?? null,
      });
    }

    // update pending queue from server
    state.pending = data.pending || state.pending;

    state.lastPrediction = await predict(currentPayload()).catch(() => null);
    renderStatus();
    renderNotifications();
    updateMap();
  } catch (err) {
    console.error(err);
    // fallback to local evaluation if backend fails
    const fallback = await (async () => {
      // reuse old local logic: evaluate via predictor per-notif
      const predictionResults = await Promise.all(
        state.pending.map(async (notif) => ({
          notif,
          result: await predict({
            ...currentPayload(),
            urgent: notif.priority === "urgent",
            timeout: state.currentIndex - (notif.created_index ?? state.currentIndex) >= QUEUE_TIMEOUT_STEPS,
            repeated_count: notif.repeat_count || 1,
            queue_age_seconds: (state.currentIndex - (notif.created_index ?? state.currentIndex)) * STEP_SECONDS,
          }),
        })),
      );
      const remaining = [];
      for (const { notif, result } of predictionResults) {
        if (["SEND", "SEND_TIMEOUT", "SEND_FALLBACK"].includes(result.decision) || notif.priority === "urgent") {
          state.delivered.push({
            ...notif,
            delivered_at: timestampNow(),
            delivery_lat: row.lat,
            delivery_lon: row.lon,
            decision: result.decision,
            reason: result.reason,
            minimal_ui: result.minimal_ui || false,
            distraction_risk: result.distraction_risk || 0,
            acceleration: result.acceleration || 0,
          });
        } else {
          remaining.push(notif);
        }
      }
      state.pending = remaining;
      return true;
    })();
    state.lastPrediction = await predict(currentPayload()).catch(() => null);
    renderStatus();
    renderNotifications();
    updateMap();
  }
}

function resetApp() {
  state.currentIndex = 0;
  state.pending = bootstrapNotifications();
  state.delivered = [];
  state.stableGoodSteps = 0;
  state.recentDropSteps = 999;
  renderNotifications();
  updateMap();
  evaluateCurrentState();
}

function queueNotification() {
  const text = document.getElementById("messageInput").value.trim();
  const priority = document.getElementById("prioritySelect").value;
  const existing = state.pending.find((item) => (item.base_text || item.text) === text && item.priority === priority);
  if (!text) return;
  if (existing) {
    existing.repeat_count = (existing.repeat_count || 1) + 1;
    existing.text = `${existing.base_text || text} (${existing.repeat_count}x)`;
  } else {
    state.pending.push({
      id: `n${state.pending.length + state.delivered.length + 1}`,
      text,
      base_text: text,
      priority,
      created_at: timestampNow(),
      created_index: state.currentIndex,
      repeat_count: 1,
    });
  }
  collapseDeferredQueue();
  renderNotifications();
  renderStatus();
}

async function init() {
  const csvText = await fetch(DATASET_URL).then((res) => res.text());
  state.route = parseCsv(csvText);
  state.pending = bootstrapNotifications();

  initMap();
  startWsSimulation();
  renderNotifications();
  await evaluateCurrentState();

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
