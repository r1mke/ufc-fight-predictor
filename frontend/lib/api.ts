import type { FighterDetail, FighterSummary, ModelsResponse, PredictRequest, PredictResponse } from "@/types";

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
