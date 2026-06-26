"use strict";

const api = (path, opts) =>
  fetch(path, opts).then((r) => {
    if (!r.ok) return r.json().then((e) => Promise.reject(e.detail || r.statusText));
    return r.json();
  });

// ---- Tabs ----
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(tab.dataset.tab).classList.add("active");
    if (tab.dataset.tab === "rankings") loadRankings();
    if (tab.dataset.tab === "simulate") loadGroups();
  });
});

// ---- Init: meta + team dropdowns ----
async function init() {
  try {
    const [health, teamsData] = await Promise.all([
      api("/api/health"),
      api("/api/teams"),
    ]);
    document.getElementById("meta").innerHTML =
      `Data through <b>${health.data_through}</b><br>` +
      `Test accuracy <b>${(health.metrics.test_accuracy * 100).toFixed(1)}%</b> · ${health.n_teams} teams`;

    const teams = teamsData.teams;
    const home = document.getElementById("home");
    const away = document.getElementById("away");
    teams.forEach((t) => {
      home.add(new Option(t, t));
      away.add(new Option(t, t));
    });
    // Sensible defaults
    setSelect(home, "Argentina");
    setSelect(away, "France");
  } catch (e) {
    document.getElementById("meta").textContent = "Failed to load model: " + e;
  }
}

function setSelect(sel, val) {
  const opt = [...sel.options].find((o) => o.value === val);
  if (opt) sel.value = val;
}

// ---- Match predictor ----
document.getElementById("predictBtn").addEventListener("click", async () => {
  const home = document.getElementById("home").value;
  const away = document.getElementById("away").value;
  const neutral = document.getElementById("neutral").checked;
  const btn = document.getElementById("predictBtn");
  if (home === away) {
    alert("Pick two different teams.");
    return;
  }
  btn.disabled = true;
  try {
    const r = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ home, away, neutral }),
    });
    showPrediction(r);
  } catch (e) {
    alert("Prediction failed: " + e);
  } finally {
    btn.disabled = false;
  }
});

function showPrediction(r) {
  const ph = r.prob_home_win, pd = r.prob_draw, pa = r.prob_away_win;
  document.getElementById("lblHome").textContent = r.home_team + " win";
  document.getElementById("lblAway").textContent = r.away_team + " win";
  document.getElementById("pHome").textContent = pct(ph);
  document.getElementById("pDraw").textContent = pct(pd);
  document.getElementById("pAway").textContent = pct(pa);
  document.getElementById("barHome").style.width = ph * 100 + "%";
  document.getElementById("barDraw").style.width = pd * 100 + "%";
  document.getElementById("barAway").style.width = pa * 100 + "%";
  document.getElementById("eloStat").textContent =
    `ELO — ${r.home_team}: ${r.elo_home}  ·  ${r.away_team}: ${r.elo_away}` +
    (r.neutral ? "  (neutral venue)" : "  (home advantage applied)");
  document.getElementById("predictResult").classList.remove("hidden");
}

const pct = (x) => (x * 100).toFixed(1) + "%";

// ---- Rankings ----
let rankingsLoaded = false;
async function loadRankings() {
  if (rankingsLoaded) return;
  const body = document.getElementById("rankBody");
  body.innerHTML = "";
  const { rankings } = await api("/api/rankings?top=40");
  rankings.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.rank}</td><td>${r.team}</td><td>${r.elo}</td>`;
    body.appendChild(tr);
  });
  rankingsLoaded = true;
}

// ---- Groups ----
let groupsLoaded = false;
async function loadGroups() {
  if (groupsLoaded) return;
  const grid = document.getElementById("groupGrid");
  grid.innerHTML = "";
  const { groups } = await api("/api/groups");
  Object.entries(groups).forEach(([name, teams]) => {
    const box = document.createElement("div");
    box.className = "groupbox";
    box.innerHTML =
      `<h4>Group ${name}</h4><ul>` +
      teams.map((t) => `<li>${t}</li>`).join("") +
      "</ul>";
    grid.appendChild(box);
  });
  groupsLoaded = true;
}

// ---- Simulator ----
document.getElementById("simBtn").addEventListener("click", async () => {
  const nSims = parseInt(document.getElementById("nSims").value, 10);
  const btn = document.getElementById("simBtn");
  const status = document.getElementById("simStatus");
  btn.disabled = true;
  status.textContent = `Running ${nSims.toLocaleString()} simulations…`;
  try {
    const t0 = performance.now();
    const r = await api("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ n_sims: nSims }),
    });
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    status.textContent = `Done in ${secs}s`;
    renderChart(r.results.slice(0, 15));
  } catch (e) {
    status.textContent = "Simulation failed: " + e;
  } finally {
    btn.disabled = false;
  }
});

function renderChart(results) {
  const chart = document.getElementById("simChart");
  chart.innerHTML = "";
  const max = Math.max(...results.map((r) => r.probability), 1);
  results.forEach((r) => {
    const row = document.createElement("div");
    row.className = "barrow";
    row.innerHTML =
      `<span class="name">${r.team}</span>` +
      `<span class="track"><span class="fill" style="width:${(r.probability / max) * 100}%"></span></span>` +
      `<span class="pct">${r.probability.toFixed(1)}%</span>`;
    chart.appendChild(row);
  });
}

init();
