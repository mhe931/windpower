// Solar views: site simulation (map, settings, result) and national overview. Talks only to /api/v1/solar/*.
import { getJson, apiUrl, ApiError } from "./api.js";
import { $, h, s, clear } from "./dom.js";
import { fmtDateTime, fmtDate, fmtNum, isoDay, parseUtc } from "./format.js";
import { drawChart, legendItem } from "./chart.js";
import { createMap } from "./map.js";
import { day, stat, setFieldError, clearFieldError, showFormError, hideFormError, watchOnce, bindTable, redraws } from "./ui.js";

const API = "/api/v1/solar";
const S = { meta: null, error: null, coverage: null, map: null, sim: null, ov: null, ovLoaded: false, simToken: 0, ovToken: 0 };
const PARAM_INPUT = { lat: "sol-lat", lon: "sol-lon", start: "sol-start", end: "sol-end", capacity_mwp: "sol-capacity" };
const MODE_TEXT = {
  hybrid: "Satellite direct and diffuse irradiance (CM SAF SARAH-3) where at least 99.9 % of the cell area has valid data in that hour, otherwise ERA5.",
  era5: "ERA5 reanalysis irradiance for every hour.",
};
// direct labels beside the line ends are short; the legend carries the full wording
const LABELS = { era5_only_mw: "ERA5-only", hybrid_mw: "Hybrid", fingrid_248_forecast_mw: "Fingrid forecast", fingrid_362_av06_small_scale_surplus_mw: "Surplus (362)" };

export function solarBanner() {
  if (!S.meta) return { text: S.error ?? "The solar artifact has not been built.", real: true, tag: "" };
  const m = S.meta;
  return { text: m.notice, real: !m.synthetic && !m.partial, tag: m.demo ? "[DEMO] " : m.synthetic ? "[SYNTHETIC] " : "" };
}

// One list item per distinct cause (short cause, detail, what it affects), not one per affected item.
function causeItems(meta, kind) {
  return (meta.unavailable.causes ?? [])
    .map((c) => ({ ...c, affects: c.affects.filter((a) => a.kind === kind) }))
    .filter((c) => c.affects.length > 0);
}

function setPartialNotices(meta) {
  for (const [id, kind] of [["sol-notice", "mode"], ["sov-notice", "series"]]) {
    const el = $(id);
    const list = $(`${id}-list`);
    clear(list);
    const items = causeItems(meta, kind);
    el.hidden = items.length === 0;
    for (const c of items) {
      list.append(h("li", {}, h("strong", {}, `${c.short}. `), `${c.detail} `, h("span", {}, `Affects: ${c.affects.map((a) => a.title).join(", ")}.`)));
    }
  }
}

// ---------------------------------------------------------------- start-up
export async function initSolar(st) {
  try {
    S.meta = await getJson(`${API}/meta`);
  } catch (err) {
    st.error = err instanceof ApiError ? err.message : "The solar data could not be loaded.";
    S.error = st.error;
    return;
  }
  st.meta = S.meta;
  S.coverage = await getJson(`${API}/coverage`).catch(() => null);
  setPartialNotices(S.meta);
  initSimulation(S.meta);
  initOverview(S.meta);
}

export function showSolarView(route) {
  const m = S.meta;
  $("footer-label").textContent = `${m.label}. ${m.statements.scope}`;
  $("footer-attribution").textContent = Object.values(m.licences).filter((v) => v && v.attribution).map((v) => v.attribution).join(" · ");
  $("footer-model").textContent = m.model_version;
  $("footer-mode").textContent = m.run_mode;
  $("footer-capacity").textContent = "n/a (solar capacity comes from Fingrid 267)";
  const f = $("footer-solar");
  f.hidden = false;
  f.textContent = `Panel ${m.panel.panel}, tilt ${m.panel.slope_deg}°, azimuth ${m.panel.azimuth_deg}°, inverter efficiency ${m.panel.inverter_efficiency}. ${m.statements.calibration}`;
  if (route === "solar-overview" && !S.ovLoaded) {
    S.ovLoaded = true;
    loadOverview({});
  } else {
    for (const fn of redraws.values()) requestAnimationFrame(fn);
  }
}

// ---------------------------------------------------------------- site simulation
function legendSwatch(kind, text) {
  const pid = `sol-legend-hatch-${kind}`;
  const svg = s("svg", { viewBox: "0 0 28 18", "aria-hidden": "true", focusable: "false", class: "map-swatch" });
  if (kind === "covered") {
    svg.append(s("defs", {}, s("pattern", { id: pid, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
      s("rect", { width: 6, height: 6, class: "hatch-bg" }), s("line", { x1: 0, y1: 0, x2: 0, y2: 6, class: "hatch-line" }))));
    svg.append(s("rect", { x: 1, y: 1, width: 26, height: 16, fill: `url(#${pid})`, class: "swatch-box" }));
  } else if (kind === "plain") svg.append(s("rect", { x: 1, y: 1, width: 26, height: 16, class: "swatch-box plain" }));
  else svg.append(s("rect", { x: 1, y: 1, width: 26, height: 16, class: "swatch-box selected" }));
  return h("li", {}, svg, text);
}

function nearestCell(g, lat, lon) {
  const step = (a, b, n) => (n > 1 ? (b - a) / (n - 1) : g.resolution_deg);
  const clamp = (v, n) => Math.max(0, Math.min(n - 1, v));
  const i = clamp(Math.round((lat - g.lat_range[0]) / step(g.lat_range[0], g.lat_range[1], g.n_lat)), g.n_lat);
  const j = clamp(Math.round((lon - g.lon_range[0]) / step(g.lon_range[0], g.lon_range[1], g.n_lon)), g.n_lon);
  return { lat: g.lat_range[0] + i * step(g.lat_range[0], g.lat_range[1], g.n_lat), lon: g.lon_range[0] + j * step(g.lon_range[0], g.lon_range[1], g.n_lon) };
}

function coverageOf(i, j) {
  const c = S.coverage;
  if (!c || !c.available) return null;
  const k = i * c.cells.n_lon + j;
  return { covered: !!c.cells.satellite_covered[k], fraction: c.cells.satellite_coverage_fraction[k] };
}

function initSimulation(meta) {
  const [first, last] = [day(meta.time_range.first_utc), day(meta.time_range.last_utc)];
  for (const id of ["sol-start", "sol-end", "sov-start", "sov-end"]) {
    $(id).min = first;
    $(id).max = last;
  }
  $("sol-range-hint").textContent = `Available period: ${fmtDate(parseUtc(meta.time_range.first_utc))} to ${fmtDate(parseUtc(meta.time_range.last_utc))} (UTC), ${fmtNum(meta.time_range.hours, 0)} hours.`;
  const [s0, s1] = lastDays(meta, 7);
  $("sol-start").value = s0;
  $("sol-end").value = s1;
  const cap = meta.limits.capacity_mwp;
  const capInput = $("sol-capacity");
  capInput.min = cap.min;
  capInput.max = cap.max;
  capInput.value = cap.default;
  $("sol-cap-hint").textContent = `Nominal (DC peak) capacity in MWp, from ${fmtNum(cap.min, 3)} to ${fmtNum(cap.max, 0)} MWp. Output scales linearly with capacity.`;

  // irradiance mode radio group: unavailable modes are disabled and say why; the first available mode is preselected
  const firstAvailable = meta.defaults.mode ?? ["hybrid", "era5"].find((m) => meta.modes[m].available) ?? null;
  for (const m of ["hybrid", "era5"]) {
    const input = $(`sol-mode-${m}`);
    const info = meta.modes[m];
    input.disabled = !info.available;
    input.checked = info.available && firstAvailable === m;
    $(`sol-mode-${m}-note`).textContent = info.available ? MODE_TEXT[m] : `Unavailable: ${info.reason_short ?? info.reason}`;
  }
  const run = $("sol-run");
  const note = $("sol-run-note");
  run.disabled = firstAvailable === null;
  note.hidden = firstAvailable !== null;
  if (firstAvailable === null) {
    const why = [...new Set(Object.values(meta.modes).map((i) => i.reason_short ?? i.reason))].join("; ");
    note.textContent = `Run simulation is disabled: no irradiance mode is available (${why}). The map and the satellite coverage can still be explored.`;
  }

  const status = $("sol-map-status");
  const setStatus = (r) => {
    const c = S.map.cellCenter(r.i, r.j);
    const cov = coverageOf(r.i, r.j);
    const covText = cov === null ? "Satellite coverage map unavailable." : cov.covered
      ? "Fully covered by SARAH-3: the hybrid mode uses satellite irradiance here."
      : "Not fully covered by SARAH-3: the hybrid mode falls back to ERA5 here (northern fallback).";
    status.textContent = `Selected cell: row ${r.i}, column ${r.j}, centre ${c.lat.toFixed(2)}°N ${c.lon.toFixed(2)}°E. ${covText}`;
  };
  S.map = createMap($("sol-map"), meta, {
    onSelect(r) {
      $("sol-lat").value = r.lat;
      $("sol-lon").value = r.lon;
      for (const id of ["sol-lat", "sol-lon"]) clearFieldError($(id));
      hideFormError("sol-errors");
      setStatus(r);
    },
    onActivate: () => $("sol-form").requestSubmit(),
  });
  const cov = S.coverage;
  const legend = $("sol-map-legend");
  clear(legend);
  if (cov && cov.available) {
    S.map.setOverlay(cov.cells.satellite_covered);
    $("sol-coverage-note").textContent = `${fmtNum(cov.summary.fully_covered_cells, 0)} of ${fmtNum(cov.summary.grid.n_cells, 0)} cells are fully covered by SARAH-3 (at least ${fmtNum(cov.threshold * 100, 1)} % of the cell area, every source pixel with daytime data).`;
    legend.append(legendSwatch("covered", "Hatched: covered by SARAH-3 (hybrid uses the satellite)"), legendSwatch("plain", "Plain: not fully covered (hybrid uses ERA5: northern fallback)"));
    $("sol-overlay").addEventListener("change", (ev) => S.map.showOverlay(ev.target.checked));
  } else {
    $("sol-overlay").disabled = true;
    $("sol-overlay").checked = false;
    $("sol-coverage-note").textContent = `Satellite coverage map unavailable: ${cov?.reason ?? "SARAH-3 was not ingested"}.`;
  }
  legend.append(legendSwatch("selected", "Thick outline: selected cell"));

  // default selection: a cell in southern Finland
  const g = meta.grid;
  const r0 = nearestCell(g, 61.0, 25.0);
  $("sol-lat").value = r0.lat;
  $("sol-lon").value = r0.lon;
  S.map.select(r0.lat, r0.lon);
  setStatus(S.map.current());

  const syncMap = () => {
    S.map.select($("sol-lat").valueAsNumber, $("sol-lon").valueAsNumber);
    const c = S.map.current();
    if (c) setStatus(c);
    else status.textContent = "That location is outside the map area.";
  };
  $("sol-lat").addEventListener("change", syncMap);
  $("sol-lon").addEventListener("change", syncMap);
  for (const el of $("sol-form").querySelectorAll("input, select")) {
    el.addEventListener("input", () => {
      clearFieldError(el);
      hideFormError("sol-errors");
    });
  }
  $("sol-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    runSimulation();
  });
  $("sol-map-help-open").addEventListener("click", () => $("sol-map-dialog").showModal());
}

function lastDays(meta, n) {
  const lastMs = parseUtc(meta.time_range.last_utc);
  const firstMs = parseUtc(meta.time_range.first_utc);
  return [isoDay(Math.max(firstMs, lastMs - (n - 1) * 24 * 3600e3)), isoDay(lastMs)];
}

function selectedMode() {
  return document.querySelector('input[name="sol_mode"]:checked')?.value ?? "";
}

async function runSimulation() {
  const form = $("sol-form");
  for (const el of form.querySelectorAll("input")) clearFieldError(el);
  hideFormError("sol-errors");
  if (!form.checkValidity()) {
    const bad = [...form.querySelectorAll("input")].filter((el) => !el.validity.valid && el.type !== "radio" && el.type !== "checkbox");
    for (const el of bad) setFieldError(el, el.validationMessage || "Please enter a valid value.");
    showFormError("sol-errors", "sol-error-text", `Please correct ${bad.length === 1 ? "the highlighted field" : "the highlighted fields"}: ${bad.map((el) => form.querySelector(`label[for="${el.id}"]`)?.textContent ?? el.name).join(", ")}.`);
    bad[0]?.focus();
    return;
  }
  const mode = selectedMode();
  if (!mode) {
    const why = [...new Set(Object.values(S.meta.modes).map((i) => i.reason_short ?? i.reason))].join("; ") || "no irradiance mode is available";
    showFormError("sol-errors", "sol-error-text", `No irradiance mode can be used: ${why}`);
    return;
  }
  const token = ++S.simToken;
  const params = { lat: $("sol-lat").value, lon: $("sol-lon").value, start: $("sol-start").value, end: $("sol-end").value, capacity_mwp: $("sol-capacity").value, mode };
  $("sol-status").textContent = "Running simulation…";
  try {
    const res = await getJson(`${API}/simulate`, params);
    if (token !== S.simToken) return;
    renderSimulation(res);
  } catch (err) {
    if (token !== S.simToken) return;
    $("sol-status").textContent = "";
    const e = err instanceof ApiError ? err : new ApiError(0, "client_error", "Something went wrong while showing the result.");
    const id = e.param === "mode" ? `sol-mode-${mode}` : PARAM_INPUT[e.param];
    if (id) setFieldError($(id), e.message);
    showFormError("sol-errors", "sol-error-text", e.message);
  }
}

function bandsOf(flags) {
  const out = [];
  let a = -1;
  flags.forEach((f, i) => {
    if (f && a < 0) a = i;
    if (!f && a >= 0) {
      out.push([a, i - 1]);
      a = -1;
    }
  });
  if (a >= 0) out.push([a, flags.length - 1]);
  return out;
}

function renderSimulation(res) {
  S.sim = res;
  const { summary: sm, cell, request: rq, data, mode } = res;
  $("sol-results").hidden = false;
  const cov = cell.satellite_covered === null ? "satellite coverage unknown" : cell.satellite_covered ? "fully covered by SARAH-3" : "not fully covered by SARAH-3";
  $("sol-cell").textContent = `${res.synthetic ? "SYNTHETIC DATA. " : ""}Cell row ${cell.lat_index}, column ${cell.lon_index}, centre ${cell.center_lat.toFixed(2)}°N ${cell.center_lon.toFixed(2)}°E, ${cell.resolution_deg}° resolution (about ${cell.extent_km.north_south} km × ${cell.extent_km.east_west} km), ${cov}. ${fmtNum(rq.capacity_mwp, 3)} MWp, ${mode.title}. Source: ${cell.source}.`;
  $("sol-cell-ew").textContent = String(cell.extent_km.east_west);
  $("sol-mode-note").textContent = mode.explanation;
  const stats = $("sol-stats");
  clear(stats);
  stats.append(
    stat("Total energy", fmtNum(sm.total_mwh, 2), "MWh"),
    stat("Mean power", fmtNum(sm.mean_mw, 3), "MW"),
    stat("Peak power", fmtNum(sm.peak_mw, 3), "MW"),
    stat("Mean capacity factor", fmtNum(sm.mean_capacity_factor * 100, 1), "%"),
    stat("Full-load hours", fmtNum(sm.full_load_hours, 1), "h"),
    stat("Hours simulated", fmtNum(sm.hours, 0), `(${fmtNum(sm.zero_output_hours, 0)} with zero output)`),
    stat("Satellite hours", fmtNum(sm.satellite_hours, 0), `of ${fmtNum(sm.hours, 0)}`),
    stat("ERA5 fallback hours", fmtNum(sm.fallback_hours, 0), `of ${fmtNum(sm.hours, 0)}`),
  );
  const t = data.time_utc.map(parseUtc);
  $("sol-caption").textContent = `${res.synthetic ? "SYNTHETIC: " : ""}Simulated hourly PV power, ${mode.name === "hybrid" ? "hybrid" : "ERA5-only"} input, ${fmtNum(rq.capacity_mwp, 3)} MWp, ${fmtDateTime(t[0])} to ${fmtDateTime(t[t.length - 1])}. Total ${fmtNum(sm.total_mwh, 2)} MWh, mean capacity factor ${fmtNum(sm.mean_capacity_factor * 100, 1)} %. Uncalibrated.`;
  const legend = $("sol-legend");
  clear(legend);
  legend.append(legendItem("swatch-pow", "Simulated power (MW, left axis)"), h("span", {}, "Capacity factor = power ÷ MWp (right axis)"));
  const gaps = bandsOf(data.irradiance_source.map((x) => x === "era5_fallback"));
  if (gaps.length) legend.append(h("span", {}, "Hatched bands = hours using ERA5 instead of the satellite"));
  const svg = $("sol-chart");
  const draw = () =>
    drawChart(svg, {
      times: t,
      series: [{ key: "pow", cls: "series-pow", label: "Power (MW)", values: data.power_mw }],
      gaps, gapHatch: true, gapClass: "band-fallback",
      yTitle: "Power (MW)",
      y2: { factor: rq.capacity_mwp, title: "Capacity factor" },
      readoutEl: $("sol-readout"),
      readout: (i) => `${fmtDateTime(t[i])}: ${fmtNum(data.power_mw[i], 4)} MW, capacity factor ${fmtNum(data.capacity_factor[i], 3)}, ${fmtNum(data.energy_mwh[i], 4)} MWh, irradiance ${data.irradiance_source[i]}${data.fallback_reason[i] ? ` (${data.fallback_reason[i]})` : ""}`,
    });
  watchOnce(svg, draw);
  draw();
  const reasons = $("sol-reasons");
  clear(reasons);
  if (mode.name === "hybrid") reasons.append(h("li", {}, `Satellite (SARAH-3): ${fmtNum(sm.satellite_hours, 0)} h`));
  else reasons.append(h("li", {}, `ERA5: ${fmtNum(sm.era5_hours, 0)} h`));
  for (const [why, n] of Object.entries(sm.fallback_reasons)) reasons.append(h("li", {}, `ERA5 fallback, ${why}: ${fmtNum(n, 0)} h — ${sm.fallback_explanations?.[why] ?? ""}`));
  const det = $("sol-table-details");
  det.open = false;
  det.querySelector("tbody").replaceChildren();
  bindTable(det, det.querySelector("tbody"), () => data.time_utc.map((u, i) => [fmtDateTime(parseUtc(u)), fmtNum(data.power_mw[i], 4), fmtNum(data.capacity_factor[i], 4), fmtNum(data.energy_mwh[i], 4), data.irradiance_source[i], data.fallback_reason[i] ?? ""]));
  $("sol-csv").href = apiUrl(`${API}/simulate.csv`, { lat: rq.lat, lon: rq.lon, start: rq.start, end: rq.end, capacity_mwp: rq.capacity_mwp, mode: rq.mode });
  $("sol-status").textContent = `Simulation complete: ${fmtNum(sm.total_mwh, 2)} MWh over ${fmtNum(sm.hours, 0)} hours, ${fmtNum(sm.satellite_hours, 0)} satellite hours and ${fmtNum(sm.fallback_hours, 0)} ERA5 fallback hours.`;
}

// ---------------------------------------------------------------- national overview
function initOverview(meta) {
  const [a, b] = [day(meta.time_range.first_utc), day(meta.time_range.last_utc)];
  $("sov-start").value = a;
  $("sov-end").value = b;
  $("sov-form").addEventListener("submit", (ev) => {
    ev.preventDefault();
    if (!$("sov-form").checkValidity()) {
      showFormError("sov-errors", "sov-error-text", "Please enter a first and last day within the available period.");
      return;
    }
    loadOverview({ start: $("sov-start").value, end: $("sov-end").value });
  });
  $("sov-all").addEventListener("click", () => {
    $("sov-start").value = a;
    $("sov-end").value = b;
    loadOverview({});
  });
  $("sov-week").addEventListener("click", () => {
    const [x, y] = lastDays(meta, 7);
    $("sov-start").value = x;
    $("sov-end").value = y;
    loadOverview({ start: x, end: y });
  });
}

async function loadOverview(params) {
  hideFormError("sov-errors");
  const token = ++S.ovToken;
  $("sov-status").textContent = "Updating overview…";
  try {
    const res = await getJson(`${API}/national`, params);
    if (token !== S.ovToken) return;
    renderOverview(res);
  } catch (err) {
    if (token !== S.ovToken) return;
    $("sov-status").textContent = "";
    showFormError("sov-errors", "sov-error-text", err instanceof ApiError ? err.message : "Something went wrong while showing the overview.");
  }
}

function tile(rec) {
  const sm = rec.summary;
  const state = rec.available ? `Available: ${fmtNum(sm.hours_with_value, 0)} of ${fmtNum(sm.hours, 0)} hours (${fmtNum(sm.coverage_pct, 1)} %)` : `Unavailable: ${rec.reason_short ?? rec.reason}`;
  return h("li", { class: rec.available ? "tile" : "tile unavailable" },
    h("strong", {}, rec.title),
    h("span", { class: "tile-state" }, rec.available ? "✔ " : "✖ ", state),
    h("small", {}, `Unit ${rec.unit}${rec.native_unit !== rec.unit ? ` (native ${rec.native_unit})` : ""}. ${rec.scope}.`));
}

function chartFor(svgId, o) {
  const svg = $(svgId);
  const draw = () => drawChart(svg, o);
  watchOnce(svg, draw);
  draw();
}

function renderOverview(res) {
  S.ov = res;
  const t = res.data.time_utc.map(parseUtc);
  const ser = res.series;
  $("sov-start").value = day(res.request.start);
  $("sov-end").value = day(res.request.end);
  const tiles = $("sov-tiles");
  clear(tiles);
  for (const rec of Object.values(ser)) tiles.append(tile(rec));
  const zoom = (a, b) => {
    const iso = (ms) => new Date(ms).toISOString().slice(0, 19) + "Z";
    loadOverview({ start: iso(a), end: iso(b) });
  };

  // chart 1: modelled national PV + Fingrid forecast
  const defs = [
    ["era5_only_mw", "series-era5", "swatch-era5", "ERA5-only model (solid line)"],
    ["hybrid_mw", "series-hybrid", "swatch-hybrid", "Hybrid SARAH-3 + ERA5 model (dashed line)"],
    ["fingrid_248_forecast_mw", "series-fc", "swatch-fc", "Fingrid forecast (not measured generation) (dotted line)"],
  ].filter(([k]) => ser[k]?.available);
  const unavailable1 = ["era5_only_mw", "hybrid_mw", "fingrid_248_forecast_mw"].filter((k) => !ser[k]?.available);
  // one short state line instead of repeating the full notices (those are in the cause list at the top and in the tiles)
  const n1 = $("sov-model-unavailable");
  n1.hidden = unavailable1.length === 0;
  n1.textContent = stateLine(ser, unavailable1, defs.length === 0);
  $("sov-note1").hidden = defs.length === 0;
  $("sov-fig1").hidden = defs.length === 0;
  if (defs.length) {
    $("sov-caption").textContent = `${res.synthetic ? "SYNTHETIC: " : ""}Modelled national PV (uncalibrated)${ser.fingrid_248_forecast_mw?.available ? " and Fingrid solar forecast (not measured generation)" : ""}, ${fmtDateTime(t[0])} to ${fmtDateTime(t[t.length - 1])}.`;
    const legend = $("sov-legend");
    clear(legend);
    for (const [, , sw, text] of defs) legend.append(legendItem(sw, text));
    chartFor("sov-chart", {
      times: t,
      series: defs.map(([k, cls]) => ({ key: k === "era5_only_mw" ? "era5" : k === "hybrid_mw" ? "hybrid" : "fc", cls, label: LABELS[k], values: ser[k].values })),
      yTitle: "Power (MW)",
      readoutEl: $("sov-readout"),
      readout: (i) => `${fmtDateTime(t[i])}: ${defs.map(([k]) => `${LABELS[k]} ${ser[k].values[i] === null ? "missing" : `${fmtNum(ser[k].values[i], 1)} MW`}`).join("; ")}`,
      onZoom: zoom,
    });
  }

  // chart 2: small-scale surplus, separate panel with its own axis
  const sur = ser.fingrid_362_av06_small_scale_surplus_mw;
  const n2 = $("sov-sur-unavailable");
  n2.hidden = !!sur?.available;
  n2.textContent = sur?.available ? "" : stateLine(ser, ["fingrid_362_av06_small_scale_surplus_mw"], true);
  $("sov-note2").hidden = !sur?.available;
  $("sov-fig2").hidden = !sur?.available;
  if (sur?.available) {
    $("sov-caption2").textContent = `${res.synthetic ? "SYNTHETIC: " : ""}Fingrid small-scale surplus, solar (AV06), kWh/h shown as MW — different scope, not total generation, ${fmtDateTime(t[0])} to ${fmtDateTime(t[t.length - 1])}.`;
    const legend = $("sov-legend2");
    clear(legend);
    legend.append(legendItem("swatch-sur", "Small-scale surplus only (kWh/h as MW) - different scope, not total generation"));
    chartFor("sov-chart2", {
      times: t,
      series: [{ key: "sur", cls: "series-sur", label: LABELS.fingrid_362_av06_small_scale_surplus_mw, values: sur.values }],
      yTitle: "Surplus (MW = kWh/h ÷ 1000)",
      readoutEl: $("sov-readout2"),
      readout: (i) => `${fmtDateTime(t[i])}: ${sur.values[i] === null ? "missing" : `${fmtNum(sur.values[i], 3)} MW (${fmtNum(sur.values_native_kwh?.[i], 1)} kWh/h)`}`,
      onZoom: zoom,
    });
  }

  // summary stats
  const stats = $("sov-stats");
  clear(stats);
  for (const rec of Object.values(ser)) {
    if (!rec.available) continue;
    const sm = rec.summary;
    if (sm.energy_gwh !== undefined) stats.append(stat(`Energy: ${rec.title}`, fmtNum(sm.energy_gwh, 3), "GWh"));
    else stats.append(stat(`Mean: ${rec.title}`, fmtNum(sm.mean, 0), "MW"));
  }

  // table + csv from the same response; with no available series there is nothing to tabulate or download
  const cols = Object.values(ser).filter((r) => r.available);
  const none = cols.length === 0;
  const det = $("sov-table-details");
  det.hidden = none;
  $("sov-stats").hidden = none;
  const empty = $("sov-more-empty");
  empty.hidden = !none;
  empty.textContent = none ? "No series are available in this run, so there is no summary, table or CSV to show." : "";
  const csv = $("sov-csv");
  csv.classList.toggle("disabled", none);
  csv.toggleAttribute("aria-disabled", none);
  if (none) {
    csv.removeAttribute("href");
    csv.removeAttribute("download");
    csv.textContent = "Download CSV: none available";
  } else {
    csv.href = apiUrl(`${API}/national.csv`, { start: res.request.start, end: res.request.end });
    csv.setAttribute("download", "");
    csv.textContent = "Download CSV of the plotted series";
    $("sov-table").tHead.replaceChildren(h("tr", {}, h("th", { scope: "col" }, "Time (UTC)"), ...cols.map((r) => h("th", { scope: "col" }, `${r.title} (${r.unit})`))));
    det.open = false;
    det.querySelector("tbody").replaceChildren();
    bindTable(det, det.querySelector("tbody"), () => res.data.time_utc.map((u, i) => [fmtDateTime(parseUtc(u)), ...cols.map((r) => (r.values[i] === null ? "missing" : fmtNum(r.values[i], 3)))]));
  }
  $("sov-status").textContent = `Overview updated: ${cols.length} of ${Object.keys(ser).length} series available.`;
}

function stateLine(ser, keys, all) {
  const names = keys.map((k) => LABELS[k] ?? ser[k].title).join(", ");
  const why = [...new Set(keys.map((k) => ser[k].reason_short ?? ser[k].reason))].join("; ");
  return `${all ? "No data to plot" : "Not plotted"}: ${names} (${why}).`;
}
