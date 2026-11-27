// Small UI helpers shared by the wind and solar views (no framework; text is only ever set through textContent / DOM nodes).
import { $, h } from "./dom.js";

export const day = (iso) => iso.slice(0, 10);
export const stat = (label, value, unit = "") => h("div", {}, h("dt", {}, label), h("dd", {}, value, unit ? h("small", {}, ` ${unit}`) : null));

export function setFieldError(input, message) {
  const err = $(`err-${input.name || input.id}`) ?? $(input.dataset.error);
  input.setAttribute("aria-invalid", "true");
  if (!err) return;
  err.textContent = message;
  err.hidden = false;
  const ids = new Set((input.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean));
  ids.add(err.id);
  input.setAttribute("aria-describedby", [...ids].join(" "));
}

export function clearFieldError(input) {
  const err = $(`err-${input.name || input.id}`) ?? $(input.dataset.error);
  input.setAttribute("aria-invalid", "false");
  if (!err) return;
  err.hidden = true;
  err.textContent = "";
  const ids = (input.getAttribute("aria-describedby") || "").split(/\s+/).filter((x) => x && x !== err.id);
  if (ids.length) input.setAttribute("aria-describedby", ids.join(" "));
  else input.removeAttribute("aria-describedby");
}

export function showFormError(boxId, textId, message) {
  $(textId).textContent = message;
  const box = $(boxId);
  box.hidden = false;
  box.focus();
}

export function hideFormError(boxId) {
  $(boxId).hidden = true;
}

/** Redraw callbacks of all charts, keyed by svg id (called when a hidden view becomes visible or the width changes). */
export const redraws = new Map();
const watched = new Set();

export function watchOnce(svg, redraw) {
  redraws.set(svg.id, redraw);
  if (watched.has(svg.id)) return;
  watched.add(svg.id);
  let width = 0;
  new ResizeObserver(() => {
    const w = Math.round(svg.getBoundingClientRect().width);
    if (w === width || w === 0) return;
    width = w;
    requestAnimationFrame(() => redraws.get(svg.id)?.());
  }).observe(svg);
}

const tableBound = new Map();
/** Fill a <details> data table lazily (on first open) from `rows()`; rebinding replaces the rows of the next fill. */
export function bindTable(details, tbody, rows) {
  tableBound.set(details.id, { rows, done: false });
  if (details.dataset.bound) return;
  details.dataset.bound = "1";
  details.addEventListener("toggle", () => {
    const b = tableBound.get(details.id);
    if (!details.open || b.done) return;
    b.done = true;
    const frag = document.createDocumentFragment();
    for (const r of b.rows()) frag.append(h("tr", {}, ...r.map((c, k) => (k === 0 ? h("th", { scope: "row" }, c) : h("td", {}, c)))));
    details.querySelector("tbody").replaceChildren(frag);
  });
}
