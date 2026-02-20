const API_BASE = "/api/v1";

let _token: string | null = localStorage.getItem("token");
let _refreshToken: string | null = localStorage.getItem("refreshToken");

function authHeaders(): HeadersInit {
  return _token ? { Authorization: `Bearer ${_token}`, "Content-Type": "application/json" } : { "Content-Type": "application/json" };
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { ...authHeaders(), ...(init.headers || {}) },
  });

  if (res.status === 401 && _refreshToken) {
    const ok = await refreshAuth();
    if (ok) return request<T>(path, init);
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

async function refreshAuth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: _refreshToken }),
    });
    if (!res.ok) throw new Error();
    const data = await res.json();
    setTokens(data.access_token, data.refresh_token);
    return true;
  } catch {
    clearTokens();
    return false;
  }
}

export function setTokens(access: string, refresh: string) {
  _token = access;
  _refreshToken = refresh;
  localStorage.setItem("token", access);
  localStorage.setItem("refreshToken", refresh);
}

export function clearTokens() {
  _token = null;
  _refreshToken = null;
  localStorage.removeItem("token");
  localStorage.removeItem("refreshToken");
}

export function isLoggedIn() {
  return !!_token;
}

// ── Auth ──
export const auth = {
  login: (username: string, password: string) =>
    request<{ access_token: string; refresh_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    }),
  register: (username: string, email: string, password: string) =>
    request<any>("/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, email, password }),
    }),
  me: () => request<any>("/auth/me"),
};

// ── Cameras ──
export interface Camera {
  id: number;
  name: string;
  source_type: string;
  source_path: string;
  location: string | null;
  status: string;
  vad_model: string;
  vad_threshold: number;
  enable_vlm: boolean;
  enable_agent: boolean;
  created_at: string;
}

export const cameras = {
  list: (status?: string) =>
    request<Camera[]>(`/cameras/${status ? `?status=${status}` : ""}`),
  get: (id: number) => request<Camera>(`/cameras/${id}`),
  create: (body: Partial<Camera>) =>
    request<Camera>("/cameras/", { method: "POST", body: JSON.stringify(body) }),
  update: (id: number, body: Partial<Camera>) =>
    request<Camera>(`/cameras/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  delete: (id: number) =>
    request<void>(`/cameras/${id}`, { method: "DELETE" }),
  start: (id: number) =>
    request<any>(`/cameras/${id}/start`, { method: "POST" }),
  stop: (id: number) =>
    request<any>(`/cameras/${id}/stop`, { method: "POST" }),
  pipelineStatus: (id: number) => request<any>(`/cameras/${id}/pipeline-status`),
  pipelineOverview: () => request<any>("/cameras/pipeline/overview"),
};

// ── Events ──
export interface Event {
  id: number;
  camera_id: number;
  timestamp: string;
  vad_score: number;
  vlm_type: string | null;
  vlm_description: string | null;
  vlm_confidence: number | null;
  agent_actions: any;
  acknowledged: boolean;
  created_at: string;
}

export const events = {
  list: (params?: Record<string, any>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<{ items: Event[]; total: number }>(`/events/${qs}`);
  },
  get: (id: number) => request<Event>(`/events/${id}`),
  ack: (id: number, note?: string) =>
    request<Event>(`/events/${id}/ack`, { method: "POST", body: JSON.stringify({ note }) }),
};

// ── Notifications ──
export const notifications = {
  status: () => request<any>("/notifications/status"),
  rules: () => request<any[]>("/notifications/rules"),
  createRule: (body: any) =>
    request<any>("/notifications/rules", { method: "POST", body: JSON.stringify(body) }),
  test: () => request<any>("/notifications/test", { method: "POST" }),
};

// ── Health ──
export const health = () => fetch("/health").then((r) => r.json());
