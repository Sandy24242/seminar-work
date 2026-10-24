export const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

async function parse(res) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error */
  }
  if (!res.ok) {
    const detail = body && (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail));
    const err = new Error(detail || `Request failed (HTTP ${res.status})`);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

function fetchWithTimeout(url, options = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(id));
}

export const getHealth = () => fetchWithTimeout(`${API_BASE}/health`, {}, 8000).then(parse);
export const getModelStatus = () => fetchWithTimeout(`${API_BASE}/models/status`, {}, 15000).then(parse);

export function predict(file, continueIfUncertain) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("continue_if_uncertain", continueIfUncertain ? "true" : "false");
  return fetchWithTimeout(`${API_BASE}/predict`, { method: "POST", body: fd }, 90000).then(parse);
}
