/** Error returned by the API (4xx JSON) or a transport failure; `.message` is safe to show to users. */
export class ApiError extends Error {
  constructor(status, code, message, param = null) {
    super(message);
    this.status = status;
    this.code = code;
    this.param = param;
  }
}

export function apiUrl(path, params = {}) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== "") q.set(k, String(v));
  const qs = q.toString();
  return qs ? `${path}?${qs}` : path;
}

export async function getJson(path, params = {}) {
  let res;
  try {
    res = await fetch(apiUrl(path, params), { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError(0, "network", "The local server could not be reached. Is it still running?");
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* not JSON */
  }
  if (!res.ok) {
    const e = body && body.error;
    throw new ApiError(res.status, e?.code ?? "http_error", e?.message ?? `The server answered with status ${res.status}.`, e?.param ?? null);
  }
  return body;
}
