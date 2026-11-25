const dateTimeUtc = new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", year: "numeric", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const dateUtc = new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", year: "numeric", month: "short", day: "numeric" });
const dayMonthUtc = new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", day: "numeric", month: "short" });
const hourUtc = new Intl.DateTimeFormat("en-GB", { timeZone: "UTC", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });

/** "3 Jul 2025, 14:00 UTC" from an ISO 8601 string or epoch ms. */
export const fmtDateTime = (t) => `${dateTimeUtc.format(new Date(t))} UTC`;
export const fmtDate = (t) => dateUtc.format(new Date(t));
export const fmtDayMonth = (t) => dayMonthUtc.format(new Date(t));
export const fmtHour = (t) => hourUtc.format(new Date(t));

const nf = (digits) => new Intl.NumberFormat("en-GB", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
const cache = {};
export function fmtNum(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  return (cache[digits] ??= nf(digits)).format(v);
}

/** ISO date (YYYY-MM-DD) of an ISO timestamp or epoch ms, in UTC. */
export const isoDay = (t) => new Date(t).toISOString().slice(0, 10);
export const parseUtc = (iso) => Date.parse(iso);
