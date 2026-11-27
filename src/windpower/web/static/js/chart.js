import { s, clear } from "./dom.js";
import { fmtDayMonth, fmtHour, fmtNum } from "./format.js";

const HOUR = 3600e3;
const STEPS = [6 * HOUR, 12 * HOUR, 24 * HOUR, 48 * HOUR, 7 * 24 * HOUR, 14 * 24 * HOUR, 30 * 24 * HOUR];

function niceCeil(v) {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= v) return m * p;
  return 10 * p;
}

function niceStep(max, target) {
  const raw = max / target;
  const p = 10 ** Math.floor(Math.log10(raw));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) return m * p;
  return 10 * p;
}

/**
 * Draw a line chart into `svg` (viewBox == pixel size, so text stays 12 px).
 * o.times: epoch ms (hourly); o.series: [{key,label,cls,values (null = missing)}];
 * o.gaps: [[firstIdx,lastIdx]] shaded as missing data; o.y2: {factor, title} adds a right axis = value / factor;
 * o.readout(i) -> text shown in o.readoutEl while hovering; o.onZoom(t0,t1) enables drag-to-zoom.
 * Series are told apart by line style (solid / dashed) and direct labels, not by colour alone.
 */
export function drawChart(svg, o) {
  clear(svg);
  const W = Math.max(280, Math.round(svg.getBoundingClientRect().width || svg.parentElement.clientWidth || 600));
  const H = Math.round(Math.min(380, Math.max(220, W * 0.45)));
  // direct labels beside the line ends when the plot stays wide enough; the margin is sized to the longest label so nothing is clipped
  const widest = Math.max(0, ...o.series.map((x) => Math.ceil(String(x.label ?? "").length * 7.8) + 12));
  const marginLabels = !o.y2 && W - 54 - widest >= 320;
  const m = { l: 54, r: o.y2 ? 58 : marginLabels ? widest : 14, t: 14, b: 40 };
  const pw = W - m.l - m.r;
  const ph = H - m.t - m.b;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const n = o.times.length;
  const t0 = o.times[0];
  const t1 = n > 1 ? o.times[n - 1] : t0 + HOUR;
  const all = o.series.flatMap((x) => x.values).filter((v) => v !== null && v !== undefined);
  const yMax = niceCeil(Math.max(...all, 0) * 1.05);
  const rawMin = Math.min(...all, 0);                     // forecasts can be slightly negative at night: keep them visible
  const step = niceStep(yMax - rawMin, Math.max(2, Math.floor(ph / 45)));
  const yMin = rawMin < 0 ? Math.floor(rawMin / step) * step : 0;
  const x = (t) => m.l + ((t - t0) / (t1 - t0)) * pw;
  const y = (v) => m.t + ph - ((v - yMin) / (yMax - yMin)) * ph;
  const g = s("g");
  svg.append(g);

  // missing-data / fallback bands (hatched when o.gapHatch, so they never rely on colour alone)
  if (o.gapHatch && (o.gaps ?? []).length) {
    const pid = `hatch-${svg.id}`;
    svg.append(s("defs", {}, s("pattern", { id: pid, width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
      s("rect", { width: 7, height: 7, class: "hatch-bg" }), s("line", { x1: 0, y1: 0, x2: 0, y2: 7, class: "hatch-line" }))));
  }
  for (const [a, b] of o.gaps ?? []) {
    const xa = x(o.times[a] - HOUR / 2);
    const xb = x(o.times[b] + HOUR / 2);
    g.append(s("rect", { class: o.gapClass ?? "gap-mark", fill: o.gapHatch ? `url(#hatch-${svg.id})` : null, x: Math.max(m.l, xa), y: m.t, width: Math.max(2, Math.min(m.l + pw, xb) - Math.max(m.l, xa)), height: ph }));
  }
  // y grid + ticks
  for (let v = yMin; v <= yMax + 1e-9; v += step) {
    g.append(s("line", { class: "gridline", x1: m.l, x2: m.l + pw, y1: y(v), y2: y(v) }));
    g.append(s("text", { class: "tick-label", x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, fmtNum(v, step < 1 ? 1 : 0)));
  }
  g.append(s("text", { class: "tick-label", x: 12, y: m.t + ph / 2, transform: `rotate(-90 12 ${m.t + ph / 2})`, "text-anchor": "middle" }, o.yTitle));
  if (o.y2) {
    const cfMax = Math.min(1, yMax / o.y2.factor);
    const cfStep = cfMax > 0.5 ? 0.25 : cfMax > 0.2 ? 0.1 : 0.05;
    for (let c = 0; c <= cfMax + 1e-9; c += cfStep) {
      g.append(s("text", { class: "tick-label", x: m.l + pw + 6, y: y(c * o.y2.factor) + 4 }, fmtNum(c, 2)));
    }
    g.append(s("text", { class: "tick-label", x: W - 10, y: m.t + ph / 2, transform: `rotate(90 ${W - 10} ${m.t + ph / 2})`, "text-anchor": "middle" }, o.y2.title));
  }
  // x ticks (UTC)
  const stepT = STEPS.find((st) => (t1 - t0) / st <= pw / 90) ?? STEPS[STEPS.length - 1];
  for (let t = Math.ceil(t0 / stepT) * stepT; t <= t1; t += stepT) {
    g.append(s("line", { class: "tick", x1: x(t), x2: x(t), y1: m.t + ph, y2: m.t + ph + 5 }));
    const label = stepT < 24 * HOUR ? `${fmtDayMonth(t)} ${fmtHour(t)}` : fmtDayMonth(t);
    g.append(s("text", { class: "tick-label", x: x(t), y: m.t + ph + 20, "text-anchor": "middle" }, label));
  }
  g.append(s("text", { class: "tick-label", x: m.l + pw / 2, y: H - 4, "text-anchor": "middle" }, "Time (UTC)"));
  g.append(s("line", { class: "axis", x1: m.l, x2: m.l, y1: m.t, y2: m.t + ph }));
  g.append(s("line", { class: "axis", x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph }));

  // series: break the line at missing values
  const labels = [];
  for (const ser of o.series) {
    let d = "";
    let pen = false;
    let last = -1;
    ser.values.forEach((v, i) => {
      if (v === null || v === undefined) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${x(o.times[i]).toFixed(1)} ${y(v).toFixed(1)}`;
      pen = true;
      last = i;
    });
    g.append(s("path", { class: `series ${ser.cls}`, d }));
    if (last >= 0) labels.push({ ser, x: x(o.times[last]), y: y(ser.values[last]) });
  }
  // direct labels, kept apart vertically
  labels.sort((a, b) => a.y - b.y);
  let prev = -Infinity;
  for (const l of labels) {
    let ly = marginLabels ? l.y + 4 : Math.max(l.y - 8, m.t + 12);
    if (ly - prev < 16) ly = prev + 16;
    prev = ly;
    if (marginLabels) g.append(s("text", { class: `label-${l.ser.key}`, x: m.l + pw + 6, y: ly }, l.ser.label));
    else g.append(s("text", { class: `halo label-${l.ser.key}`, x: Math.min(l.x, m.l + pw - 2), y: ly, "text-anchor": "end" }, l.ser.label));
  }

  // interaction: hover readout and drag-to-zoom
  const cursor = s("line", { class: "cursor", y1: m.t, y2: m.t + ph, visibility: "hidden" });
  const brush = s("rect", { class: "brush", y: m.t, height: ph, visibility: "hidden" });
  const dots = o.series.map((ser) => s("circle", { class: `dot ${ser.cls}`, r: 4, visibility: "hidden" }));
  const overlay = s("rect", { class: "overlay", x: m.l, y: m.t, width: pw, height: ph });
  svg.append(cursor, brush, ...dots, overlay);
  const toIndex = (ev) => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    const t = t0 + ((px - m.l) / pw) * (t1 - t0);
    return Math.max(0, Math.min(n - 1, Math.round((t - t0) / HOUR)));
  };
  const px = (ev) => {
    const r = svg.getBoundingClientRect();
    return Math.max(m.l, Math.min(m.l + pw, ((ev.clientX - r.left) / r.width) * W));
  };
  const show = (i) => {
    const vis = i === null ? "hidden" : "visible";
    cursor.setAttribute("visibility", vis);
    dots.forEach((d, k) => {
      const v = i === null ? null : o.series[k].values[i];
      d.setAttribute("visibility", v === null || v === undefined ? "hidden" : vis);
      if (v !== null && v !== undefined) {
        d.setAttribute("cx", x(o.times[i]));
        d.setAttribute("cy", y(v));
      }
    });
    if (i !== null) cursor.setAttribute("x1", x(o.times[i])), cursor.setAttribute("x2", x(o.times[i]));
    if (o.readoutEl) o.readoutEl.textContent = i === null ? "" : o.readout(i);
  };
  let dragFrom = null;
  overlay.addEventListener("pointermove", (ev) => {
    show(toIndex(ev));
    if (dragFrom !== null) {
      const a = Math.min(dragFrom, px(ev));
      brush.setAttribute("x", a);
      brush.setAttribute("width", Math.abs(px(ev) - dragFrom));
      brush.setAttribute("visibility", "visible");
    }
  });
  overlay.addEventListener("pointerleave", () => {
    if (dragFrom === null) show(null);
  });
  if (o.onZoom) {
    overlay.addEventListener("pointerdown", (ev) => {
      dragFrom = px(ev);
      overlay.setPointerCapture(ev.pointerId);
    });
    overlay.addEventListener("pointerup", (ev) => {
      if (dragFrom === null) return;
      const a = dragFrom;
      const b = px(ev);
      dragFrom = null;
      brush.setAttribute("visibility", "hidden");
      if (Math.abs(b - a) < 8) return;
      const ta = t0 + ((Math.min(a, b) - m.l) / pw) * (t1 - t0);
      const tb = t0 + ((Math.max(a, b) - m.l) / pw) * (t1 - t0);
      o.onZoom(Math.round(ta / HOUR) * HOUR, Math.round(tb / HOUR) * HOUR);
    });
    overlay.addEventListener("pointercancel", () => {
      dragFrom = null;
      brush.setAttribute("visibility", "hidden");
    });
  }
}

/** Legend entry with a line sample whose dash pattern matches the series (so it does not rely on colour). */
export function legendItem(cls, text) {
  const wrap = document.createElement("span");
  const svg = s("svg", { viewBox: "0 0 52 14", "aria-hidden": "true", focusable: "false" }, s("line", { class: cls, x1: 2, x2: 50, y1: 7, y2: 7 }));
  wrap.append(svg, text);
  return wrap;
}
