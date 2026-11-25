const SVG_NS = "http://www.w3.org/2000/svg";

/** Create an HTML element: h("p", {class: "x"}, "text", child). Text is always set as text (never HTML). */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined && v !== false) el.setAttribute(k, v === true ? "" : String(v));
  el.append(...children.filter((c) => c !== null && c !== undefined));
  return el;
}

/** Create an SVG element. */
export function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) el.setAttribute(k, String(v));
  el.append(...children.filter((c) => c !== null && c !== undefined));
  return el;
}

export const $ = (id) => document.getElementById(id);

export function clear(el) {
  el.replaceChildren();
}
