import { s, clear } from "./dom.js";

const SCALE = 100; // px per degree of latitude in the SVG coordinate system

/**
 * Finland map: coarse outline + ERA5 grid + selected cell, as inline SVG (equirectangular, longitude scaled by cos(latitude)).
 * Click/tap selects a point; with the container focused, arrow keys move the selection by one grid cell and Enter activates.
 */
export function createMap(container, meta, { onSelect, onActivate }) {
  const { lat_range: latR, lon_range: lonR, resolution_deg: res } = meta.grid;
  const nLat = meta.grid.n_lat;
  const nLon = meta.grid.n_lon;
  const lats = Array.from({ length: nLat }, (_, i) => latR[0] + (i * (latR[1] - latR[0])) / Math.max(1, nLat - 1));
  const lons = Array.from({ length: nLon }, (_, j) => lonR[0] + (j * (lonR[1] - lonR[0])) / Math.max(1, nLon - 1));
  const kx = Math.cos(((latR[0] + latR[1]) / 2) * (Math.PI / 180));
  const west = lons[0] - res / 2;
  const north = lats[nLat - 1] + res / 2;
  const W = nLon * res * kx * SCALE;
  const H = nLat * res * SCALE;
  const X = (lon) => (lon - west) * kx * SCALE;
  const Y = (lat) => (north - lat) * SCALE;

  const svg = s("svg", { viewBox: `0 0 ${W.toFixed(1)} ${H.toFixed(1)}`, "aria-hidden": "true", focusable: "false" });
  // grid lines
  let d = "";
  for (let j = 0; j <= nLon; j++) d += `M${X(west + j * res).toFixed(1)} 0V${H.toFixed(1)}`;
  for (let i = 0; i <= nLat; i++) d += `M0 ${(i * res * SCALE).toFixed(1)}H${W.toFixed(1)}`;
  svg.append(s("path", { class: "grid-line", d }));
  for (const ring of Object.values(meta.region.outlines_lonlat)) {
    svg.append(s("polygon", { class: "outline", points: ring.map(([lo, la]) => `${X(lo).toFixed(1)},${Y(la).toFixed(1)}`).join(" ") }));
  }
  // coordinate labels every 2 degrees
  for (let la = Math.ceil(lats[0] / 2) * 2; la < north; la += 2) svg.append(s("text", { class: "axis-label", x: 3, y: Y(la) - 2 }, `${la}°N`));
  for (let lo = Math.ceil(lons[0] / 2) * 2; lo < lons[nLon - 1]; lo += 2) svg.append(s("text", { class: "axis-label", x: X(lo) + 2, y: H - 3 }, `${lo}°E`));

  const ring = s("rect", { class: "cell-sel-ring", visibility: "hidden" });
  const sel = s("rect", { class: "cell-sel", visibility: "hidden" });
  const dot = s("circle", { class: "marker", r: 3.5, visibility: "hidden" });
  svg.append(ring, sel, dot);
  clear(container);
  container.append(svg);

  let cur = null; // {i, j}
  const nearest = (arr, v) => arr.reduce((b, x, k) => (Math.abs(x - v) < Math.abs(arr[b] - v) ? k : b), 0);
  function draw(i, j, lat, lon) {
    cur = { i, j };
    for (const r of [ring, sel]) {
      r.setAttribute("x", X(lons[j] - res / 2));
      r.setAttribute("y", Y(lats[i] + res / 2));
      r.setAttribute("width", res * kx * SCALE);
      r.setAttribute("height", res * SCALE);
      r.setAttribute("visibility", "visible");
    }
    dot.setAttribute("cx", X(lon ?? lons[j]));
    dot.setAttribute("cy", Y(lat ?? lats[i]));
    dot.setAttribute("visibility", "visible");
  }
  function inside(lat, lon) {
    return lat >= lats[0] - res / 2 && lat <= north && lon >= west && lon <= lons[nLon - 1] + res / 2;
  }
  function hide() {
    cur = null;
    for (const e of [ring, sel, dot]) e.setAttribute("visibility", "hidden");
  }

  svg.addEventListener("click", (ev) => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    const py = ((ev.clientY - r.top) / r.height) * H;
    const lon = west + px / (kx * SCALE);
    const lat = north - py / SCALE;
    if (!inside(lat, lon)) return;
    const i = nearest(lats, lat);
    const j = nearest(lons, lon);
    const la = Math.round(lat * 1e4) / 1e4;
    const lo = Math.round(lon * 1e4) / 1e4;
    draw(i, j, la, lo);
    onSelect({ lat: la, lon: lo, i, j, via: "pointer" });
  });

  container.addEventListener("keydown", (ev) => {
    const moves = { ArrowUp: [1, 0], ArrowDown: [-1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    if (ev.key === "Enter") {
      ev.preventDefault();
      onActivate?.();
      return;
    }
    const mv = moves[ev.key];
    if (!mv || ev.altKey || ev.ctrlKey || ev.metaKey) return;
    ev.preventDefault();
    const base = cur ?? { i: Math.floor(nLat / 2), j: Math.floor(nLon / 2) };
    const i = Math.max(0, Math.min(nLat - 1, base.i + (cur ? mv[0] : 0)));
    const j = Math.max(0, Math.min(nLon - 1, base.j + (cur ? mv[1] : 0)));
    draw(i, j);
    onSelect({ lat: lats[i], lon: lons[j], i, j, via: "keyboard" });
  });

  return {
    /** Show the cell containing (lat, lon) without notifying (used when the form fields change). */
    select(lat, lon) {
      if (!Number.isFinite(lat) || !Number.isFinite(lon) || !inside(lat, lon)) return hide();
      draw(nearest(lats, lat), nearest(lons, lon), lat, lon);
    },
    cellCenter: (i, j) => ({ lat: lats[i], lon: lons[j] }),
    current: () => cur,
  };
}
