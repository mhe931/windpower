import { getJson, apiUrl, ApiError } from "./api.js";
import { $, h, clear } from "./dom.js";
import { fmtDateTime, fmtDate, fmtNum, isoDay, parseUtc } from "./format.js";
import { drawChart, legendItem } from "./chart.js";
import { createMap } from "./map.js";
import { day, stat, setFieldError, clearFieldError, showFormError, hideFormError, redraws, watchOnce, bindTable } from "./ui.js";
import { initSolar, showSolarView, solarBanner } from "./solar.js";

const API = "/api/v1";
const PARAM_INPUT = { lat: "lat", lon: "lon", start: "start", end: "end", archetype: "archetype", capacity_mw: "capacity" };
const state = { meta: null, solar: { meta: null }, windError: null, archetypes: new Map(), map: null, sim: null, ov: null, ovLoaded: false, redraws, simToken: 0, ovToken: 0 };

// ---------------------------------------------------------------- page chrome
function applyMeta(meta) {
  const banner = $("data-banner");
  $("banner-text").textContent = meta.notice;
  banner.classList.toggle("real", !meta.synthetic);
  document.title = `${meta.demo ? "[DEMO] " : meta.synthetic ? "[SYNTHETIC] " : ""}Finland wind and solar simulator`;
  $("footer-label").textContent = `${meta.label}. ${meta.statements.real_data}`;
  $("footer-attribution").textContent = meta.attribution ?? "";
  $("footer-model").textContent = meta.model_version;
  $("footer-mode").textContent = meta.run_mode;
  $("cell-ew").textContent = String(meta.grid.cell_km_at_mid_latitude.east_west);
}

function initDateInputs(meta) {
  const first = day(meta.time_range.first_utc);
  const last = day(meta.time_range.last_utc);
  for (const id of ["start", "end", "ov-start", "ov-end"]) {
    $(id).min = first;
    $(id).max = last;
  }
  $("range-hint").textContent = `Available period: ${fmtDate(parseUtc(meta.time_range.first_utc))} to ${fmtDate(parseUtc(meta.time_range.last_utc))} (UTC), ${fmtNum(meta.time_range.hours, 0)} hours.`;
}

function lastDays(meta, n) {
  const lastMs = parseUtc(meta.time_range.last_utc);
  const firstMs = parseUtc(meta.time_range.first_utc);
  return [isoDay(Math.max(firstMs, lastMs - (n - 1) * 24 * 3600e3)), isoDay(lastMs)];
}

// ---------------------------------------------------------------- view 1: site simulation
function initSimulation(meta, archetypes) {
  const sel = $("archetype");
  for (const a of archetypes.archetypes) {
    state.archetypes.set(a.name, a);
    sel.append(h("option", { value: a.name }, `${a.name} (hub ${fmtNum(a.hub_height_m, 0)} m, ${fmtNum(a.specific_power_wm2, 0)} W/m²)`));
  }
  sel.value = archetypes.default;
  const describe = () => {
    const a = state.archetypes.get(sel.value);
    $("arch-desc").textContent = `Mean hub height ${fmtNum(a.hub_height_m, 0)} m, specific power ${fmtNum(a.specific_power_wm2, 0)} W/m², mean rated size ${fmtNum(a.mean_rated_mw, 1)} MW. In the modelled fleet: ${fmtNum(a.turbines, 0)} turbines, ${fmtNum(a.fleet_installed_mw, 1)} MW.`;
  };
  sel.addEventListener("change", describe);
  describe();

  const cap = meta.limits.capacity_mw;
  const capInput = $("capacity");
  capInput.min = cap.min;
  capInput.max = cap.max;
  capInput.value = cap.default;
  $("cap-hint").textContent = `Any value from ${fmtNum(cap.min, 2)} to ${fmtNum(cap.max, 0)} MW. Output scales linearly with capacity.`;

  const [s0, s1] = lastDays(meta, 7);
  $("start").value = s0;
  $("end").value = s1;

  const status = $("map-status");
  const setStatus = (r) => {
    const c = state.map.cellCenter(r.i, r.j);
    status.textContent = `Selected cell: row ${r.i}, column ${r.j}, centre ${c.lat.toFixed(2)}°N ${c.lon.toFixed(2)}°E. The result is the average over this whole cell.`;
  };
  state.map = createMap($("map"), meta, {
    onSelect(r) {
      $("lat").value = r.lat;
      $("lon").value = r.lon;
      for (const id of ["lat", "lon"]) clearFieldError($(id));
      hideFormError("sim-errors");
      setStatus(r);
    },
    onActivate: () => $("sim-form").requestSubmit(),
  });
  // default selection: centre of the grid
  const g = meta.grid;
  const lat0 = (g.lat_range[0] + g.lat_range[1]) / 2;
  const lon0 = (g.lon_range[0] + g.lon_range[1]) / 2;
  const r0 = nearestCell(g, lat0, lon0);
  $("lat").value = r0.lat;
  $("lon").value = r0.lon;
  state.map.select(r0.lat, r0.lon);
  setStatus(state.map.current());

  const syncMap = () => {
    const lat = $("lat").valueAsNumber;
    const lon = $("lon").valueAsNumber;
    state.map.select(lat, lon);
    const c = state.map.current();
    status.textContent = c ? "" : "That location is outside the map area.";
    if (c) setStatus(c);
  };
  $("lat").addEventListener("change", syncMap);
  $("lon").addEventListener("change", syncMap);
  for (const el of $("sim-form").querySelectorAll("input, select")) {
    el.addEventListener("input", () => {
      clearFieldError(el);
      hideFormError("sim-errors");
    });
  }
  $("sim-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    runSimulation();
  });
  $("map-help-open").addEventListener("click", () => $("map-dialog").showModal());
}

function nearestCell(g, lat, lon) {
  const step = (a, b, n) => (n > 1 ? (b - a) / (n - 1) : g.resolution_deg);
  const i = Math.round((lat - g.lat_range[0]) / step(g.lat_range[0], g.lat_range[1], g.n_lat));
  const j = Math.round((lon - g.lon_range[0]) / step(g.lon_range[0], g.lon_range[1], g.n_lon));
  return { lat: g.lat_range[0] + i * step(g.lat_range[0], g.lat_range[1], g.n_lat), lon: g.lon_range[0] + j * step(g.lon_range[0], g.lon_range[1], g.n_lon) };
}

async function runSimulation() {
  const form = $("sim-form");
  for (const el of form.querySelectorAll("input, select")) clearFieldError(el);
  hideFormError("sim-errors");
  if (!form.checkValidity()) {
    const bad = [...form.querySelectorAll("input")].filter((el) => !el.validity.valid);
    for (const el of bad) setFieldError(el, el.validationMessage || "Please enter a valid value.");
    showFormError("sim-errors", "sim-error-text", `Please correct ${bad.length === 1 ? "the highlighted field" : "the highlighted fields"}: ${bad.map((el) => form.querySelector(`label[for="${el.id}"]`)?.textContent ?? el.name).join(", ")}.`);
    bad[0].focus();
    return;
  }
  const token = ++state.simToken;
  const params = { lat: $("lat").value, lon: $("lon").value, start: $("start").value, end: $("end").value, archetype: $("archetype").value, capacity_mw: $("capacity").value };
  $("sim-status").textContent = "Running simulation…";
  try {
    const res = await getJson(`${API}/simulate`, params);
    if (token !== state.simToken) return;
    renderSimulation(res);
  } catch (err) {
    if (token !== state.simToken) return;
    $("sim-status").textContent = "";
    const e = err instanceof ApiError ? err : new ApiError(0, "client_error", "Something went wrong while showing the result.");
    const inputId = PARAM_INPUT[e.param];
    if (inputId) setFieldError($(inputId), e.message);
    showFormError("sim-errors", "sim-error-text", e.message);
  }
}

function renderSimulation(res) {
  state.sim = res;
  const { summary: sm, cell, archetype: a, request: rq, data } = res;
  $("sim-results").hidden = false;
  $("sim-cell").textContent = `${res.synthetic ? "SYNTHETIC DATA. " : ""}Cell row ${cell.lat_index}, column ${cell.lon_index}, centre ${cell.center_lat.toFixed(2)}°N ${cell.center_lon.toFixed(2)}°E, ${cell.resolution_deg}° resolution (about ${cell.extent_km.north_south} km × ${cell.extent_km.east_west} km). Archetype ${a.name}, hub ${fmtNum(a.hub_height_m, 0)} m, ${fmtNum(rq.capacity_mw, 2)} MW installed. Source: ${cell.source}.`;
  $("cell-ew").textContent = String(cell.extent_km.east_west);
  const stats = $("sim-stats");
  clear(stats);
  stats.append(
    stat("Total energy", fmtNum(sm.total_mwh, 1), "MWh"),
    stat("Mean power", fmtNum(sm.mean_mw, 2), "MW"),
    stat("Peak power", fmtNum(sm.peak_mw, 2), "MW"),
    stat("Mean capacity factor", fmtNum(sm.mean_capacity_factor * 100, 1), "%"),
    stat("Full-load hours", fmtNum(sm.full_load_hours, 1), "h"),
    stat("Hours simulated", fmtNum(sm.hours, 0), `(${fmtNum(sm.zero_output_hours, 0)} with zero output)`),
  );
  const t = data.time_utc.map(parseUtc);
  $("sim-caption").textContent = `${res.synthetic ? "SYNTHETIC: " : ""}Simulated hourly power, ${a.name}, ${fmtNum(rq.capacity_mw, 2)} MW, ${fmtDateTime(t[0])} to ${fmtDateTime(t[t.length - 1])}. Total ${fmtNum(sm.total_mwh, 1)} MWh, mean capacity factor ${fmtNum(sm.mean_capacity_factor * 100, 1)} %. Uncalibrated.`;
  const legend = $("sim-legend");
  clear(legend);
  legend.append(legendItem("swatch-theo", "Simulated power (MW, left axis)"), h("span", {}, "Capacity factor = power ÷ installed capacity (right axis)"));
  const svg = $("sim-chart");
  const draw = () =>
    drawChart(svg, {
      times: t,
      series: [{ key: "power", cls: "series-power", label: "Power (MW)", values: data.power_mw }],
      yTitle: "Power (MW)",
      y2: { factor: rq.capacity_mw, title: "Capacity factor" },
      readoutEl: $("sim-readout"),
      readout: (i) => `${fmtDateTime(t[i])}: ${fmtNum(data.power_mw[i], 3)} MW, capacity factor ${fmtNum(data.capacity_factor[i], 3)}, ${fmtNum(data.energy_mwh[i], 3)} MWh`,
    });
  watchOnce(svg, draw);
  draw();
  // data table is built lazily from the same response
  const tbody = $("sim-table").tBodies[0];
  tbody.replaceChildren();
  const det = $("sim-table-details");
  det.open = false;
  bindTable(det, tbody, () => data.time_utc.map((u, i) => [fmtDateTime(parseUtc(u)), fmtNum(data.power_mw[i], 3), fmtNum(data.capacity_factor[i], 4), fmtNum(data.energy_mwh[i], 3)]));
  $("sim-csv").href = apiUrl(`${API}/simulate.csv`, { lat: rq.lat, lon: rq.lon, start: rq.start, end: rq.end, archetype: rq.archetype, capacity_mw: rq.capacity_mw });
  $("sim-status").textContent = `Simulation complete: ${fmtNum(sm.total_mwh, 1)} MWh over ${fmtNum(sm.hours, 0)} hours, mean capacity factor ${fmtNum(sm.mean_capacity_factor * 100, 1)} percent.`;
}

  
// ---------------------------------------------------------------- view 2: national overview
function initOverview(meta) {
  const [s0, s1] = [day(meta.time_range.first_utc), day(meta.time_range.last_utc)];
  $("ov-start").value = s0;
  $("ov-end").value = s1;
  $("ov-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    if (!$("ov-form").checkValidity()) {
      showFormError("ov-errors", "ov-error-text", "Please enter a first and last day within the available period.");
      return;
    }
    loadOverview({ start: $("ov-start").value, end: $("ov-end").value });
  });
  $("ov-all").addEventListener("click", () => {
    $("ov-start").value = s0;
    $("ov-end").value = s1;
    loadOverview({});
  });
  $("ov-week").addEventListener("click", () => {
    const [a, b] = lastDays(meta, 7);
    $("ov-start").value = a;
    $("ov-end").value = b;
    loadOverview({ start: a, end: b });
  });
}

async function loadOverview(params) {
  hideFormError("ov-errors");
  const token = ++state.ovToken;
  $("ov-status").textContent = "Updating overview…";
  try {
    const res = await getJson(`${API}/national`, params);
    if (token !== state.ovToken) return;
    renderOverview(res);
  } catch (err) {
    if (token !== state.ovToken) return;
    $("ov-status").textContent = "";
    showFormError("ov-errors", "ov-error-text", err instanceof ApiError ? err.message : "Something went wrong while showing the overview.");
  }
}

function gapsOf(avail) {
  const out = [];
  let start = -1;
  avail.forEach((a, i) => {
    if (!a && start < 0) start = i;
    if (a && start >= 0) {
      out.push([start, i - 1]);
      start = -1;
    }
  });
  if (start >= 0) out.push([start, avail.length - 1]);
  return out;
}

function renderOverview(res) {
  state.ov = res;
  const { data, summary: sm } = res;
  const t = data.time_utc.map(parseUtc);
  $("ov-start").value = day(res.request.start);
  $("ov-end").value = day(res.request.end);
  $("ov-caption").textContent = `${res.synthetic ? "SYNTHETIC: " : ""}Theoretical (simulated, uncalibrated) and observed (Fingrid) national wind power, ${fmtDateTime(t[0])} to ${fmtDateTime(t[t.length - 1])}. ${fmtNum(sm.observed_hours, 0)} of ${fmtNum(sm.hours, 0)} hours have an observed value.`;
  const legend = $("ov-legend");
  clear(legend);
  legend.append(
    legendItem("swatch-theo", "Theoretical, solid line (simulated, uncalibrated)"),
    legendItem("swatch-obs", "Observed, dashed line (Fingrid, may be curtailed)"),
    h("span", {}, "Shaded band = no observed value"),
  );
  const gaps = gapsOf(data.observed_available);
  const svg = $("ov-chart");
  const draw = () =>
    drawChart(svg, {
      times: t,
      series: [
        { key: "theo", cls: "series-theo", label: "Theoretical", values: data.theoretical_mw },
        { key: "obs", cls: "series-obs", label: "Observed", values: data.observed_mw },
      ],
      gaps,
      yTitle: "Power (MW)",
      readoutEl: $("ov-readout"),
      readout: (i) => `${fmtDateTime(t[i])}: theoretical ${fmtNum(data.theoretical_mw[i], 1)} MW; observed ${data.observed_mw[i] === null ? "missing" : `${fmtNum(data.observed_mw[i], 1)} MW`}`,
      onZoom: (a, b) => {
        const iso = (ms) => new Date(ms).toISOString().slice(0, 19) + "Z";
        loadOverview({ start: iso(a), end: iso(b) });
      },
    });
  watchOnce(svg, draw);
  draw();
  const stats = $("ov-stats");
  clear(stats);
  stats.append(
    stat("Hours in period", fmtNum(sm.hours, 0)),
    stat("Observed hours", fmtNum(sm.observed_hours, 0), `(${fmtNum(sm.observed_pct, 1)} %)`),
    stat("Missing observed hours", fmtNum(sm.missing_hours, 0)),
    stat("Mean theoretical", fmtNum(sm.theoretical_mean_mw, 0), "MW"),
    stat("Mean observed", fmtNum(sm.observed_mean_mw, 0), "MW"),
    stat("Mean theoretical, observed hours only", fmtNum(sm.theoretical_mean_mw_on_observed_hours, 0), "MW"),
    stat("Peak theoretical / observed", `${fmtNum(sm.theoretical_peak_mw, 0)} / ${fmtNum(sm.observed_peak_mw, 0)}`, "MW"),
    stat("Installed fleet capacity", fmtNum(res.installed_mw, 0), "MW"),
  );
  $("footer-capacity").textContent = `${fmtNum(res.installed_mw, 1)} MW`;
  $("ov-missing-summary").textContent = sm.missing_hours === 0 ? "No missing observed hours in this period." : `${fmtNum(sm.missing_hours, 0)} of ${fmtNum(sm.hours, 0)} hours (${fmtNum(100 - sm.observed_pct, 1)} %) have no observed value; they are left blank, not filled in. Periods:`;
  const list = $("ov-missing-list");
  clear(list);
  for (const r of res.missing_ranges.slice(0, 200)) list.append(h("li", {}, `${fmtDateTime(parseUtc(r.start))}${r.hours > 1 ? ` to ${fmtDateTime(parseUtc(r.end))}` : ""} (${r.hours} ${r.hours === 1 ? "hour" : "hours"})`));
  const det = $("ov-table-details");
  det.open = false;
  det.querySelector("tbody").replaceChildren();
  bindTable(det, det.querySelector("tbody"), () => data.time_utc.map((u, i) => [fmtDateTime(parseUtc(u)), fmtNum(data.theoretical_mw[i], 1), data.observed_mw[i] === null ? "missing" : fmtNum(data.observed_mw[i], 1)]));
  $("ov-csv").href = apiUrl(`${API}/national.csv`, { start: res.request.start, end: res.request.end });
  $("ov-status").textContent = `Overview updated: ${fmtNum(sm.observed_hours, 0)} of ${fmtNum(sm.hours, 0)} hours observed.`;
}

// ---------------------------------------------------------------- routing and start-up
const WIND_VIEWS = ["simulate", "overview"];
const SOLAR_VIEWS = ["solar-simulate", "solar-overview"];

function currentRoute() {
  const hash = location.hash.slice(1);
  if ([...WIND_VIEWS, ...SOLAR_VIEWS].includes(hash)) return hash;
  return state.meta || !state.solar.meta ? "simulate" : "solar-simulate"; // a solar-only server opens on Solar
}

function setBanner(text, real) {
  $("banner-text").textContent = text;
  $("data-banner").classList.toggle("real", real);
}

function showView() {
  const route = currentRoute();
  const isSolar = route.startsWith("solar-");
  for (const v of [...WIND_VIEWS, ...SOLAR_VIEWS]) {
    const ok = isSolar ? state.solar.meta : state.meta;
    $(`view-${v}`).hidden = v !== route || !ok;
    const link = document.getElementById(`nav-${v}`);
    if (v === route) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  $("tabs-wind").hidden = isSolar;
  $("tabs-solar").hidden = !isSolar;
  for (const [id, on] of [["nav-wind", !isSolar], ["nav-solar", isSolar]]) {
    if (on) $(id).setAttribute("aria-current", "true");
    else $(id).removeAttribute("aria-current");
  }
  $("wind-missing").hidden = isSolar || !!state.meta;
  $("solar-missing").hidden = !isSolar || !!state.solar.meta;
  if (isSolar) {
    const b = solarBanner();
    setBanner(b.text, b.real);
    document.title = `${b.tag}Finland solar simulator`;
    if (state.solar.meta) showSolarView(route);
    return;
  }
  if (!state.meta) {
    setBanner(state.windError ?? "The wind artifact has not been built.", true);
    return;
  }
  applyMeta(state.meta);
  $("footer-solar").hidden = true;
  if (route === "overview" && !state.ovLoaded) {
    state.ovLoaded = true;
    loadOverview({});
  } else {
    for (const fn of state.redraws.values()) requestAnimationFrame(fn);
  }
}

async function main() {
  state.solar = { meta: null };
  const [wind, solar] = await Promise.allSettled([Promise.all([getJson(`${API}/meta`), getJson(`${API}/archetypes`)]), initSolar(state.solar)]);
  if (wind.status === "fulfilled") {
    const [meta, archetypes] = wind.value;
    state.meta = meta;
    applyMeta(meta);
    initDateInputs(meta);
    initSimulation(meta, archetypes);
    initOverview(meta);
  } else {
    const err = wind.reason;
    state.windError = err instanceof ApiError ? err.message : "The wind data could not be loaded.";
    $("wind-missing-text").textContent = state.windError;
  }
  if (solar.status === "rejected") $("solar-missing-text").textContent = "The solar data could not be loaded.";
  else if (!state.solar.meta) $("solar-missing-text").textContent = state.solar.error ?? "The solar artifact has not been built.";
  window.addEventListener("hashchange", showView);
  showView();
}

main();
