import type {
  FighterDetail,
  FighterSummary,
  ModelsResponse,
  ModelVersionsResponse,
  PendingFight,
  PendingFightCreate,
  PendingFightUpdate,
  PredictRequest,
  PredictResponse,
  RetrainStatus,
  ScrapeStatus,
} from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${res.status}`);
  }
  return res.json();
}

function adminRequest<T>(path: string, adminToken: string, init?: RequestInit): Promise<T> {
  return request(path, { ...init, headers: { "Admin-Token": adminToken, ...init?.headers } });
}

export function searchFighters(query: string, limit = 20): Promise<FighterSummary[]> {
  const params = new URLSearchParams({ search: query, limit: String(limit) });
  return request(`/api/fighters?${params}`);
}

export function getFighter(fighterId: string): Promise<FighterDetail> {
  return request(`/api/fighters/${fighterId}`);
}

export function getModels(): Promise<ModelsResponse> {
  return request(`/api/models`);
}

export function predictFight(body: PredictRequest): Promise<PredictResponse> {
  return request(`/api/predict`, { method: "POST", body: JSON.stringify(body) });
}

// --- Admin: scraping / review queue / retraining ---------------------------

export function triggerScrape(adminToken: string): Promise<ScrapeStatus> {
  return adminRequest(`/api/admin/scrape`, adminToken, { method: "POST" });
}

export function getScrapeStatus(adminToken: string): Promise<ScrapeStatus> {
  return adminRequest(`/api/admin/scrape/status`, adminToken);
}

export function getPendingFights(adminToken: string): Promise<PendingFight[]> {
  return adminRequest(`/api/admin/pending`, adminToken);
}

export function createPendingFight(adminToken: string, payload: PendingFightCreate): Promise<PendingFight> {
  return adminRequest(`/api/admin/pending`, adminToken, { method: "POST", body: JSON.stringify(payload) });
}

export function updatePendingFight(
  adminToken: string,
  id: string,
  patch: PendingFightUpdate
): Promise<PendingFight> {
  return adminRequest(`/api/admin/pending/${id}`, adminToken, { method: "PATCH", body: JSON.stringify(patch) });
}

export function deletePendingFight(adminToken: string, id: string): Promise<PendingFight> {
  return adminRequest(`/api/admin/pending/${id}`, adminToken, { method: "DELETE" });
}

export function triggerRetrain(adminToken: string, pendingIds: string[]): Promise<RetrainStatus> {
  return adminRequest(`/api/admin/retrain`, adminToken, {
    method: "POST",
    body: JSON.stringify({ pending_ids: pendingIds }),
  });
}

export function getRetrainStatus(adminToken: string): Promise<RetrainStatus> {
  return adminRequest(`/api/admin/retrain/status`, adminToken);
}

export function getModelVersions(adminToken: string): Promise<ModelVersionsResponse> {
  return adminRequest(`/api/admin/model-versions`, adminToken);
}

export function restoreModelVersion(
  adminToken: string,
  versionId: string
): Promise<{ status: string; version_id: string }> {
  return adminRequest(`/api/admin/model-versions/${versionId}/restore`, adminToken, { method: "POST" });
}
