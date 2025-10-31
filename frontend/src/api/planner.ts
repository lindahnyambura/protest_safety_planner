// frontend/src/api/planner.ts
// Handles communication with the Risk-Aware Route Planner FastAPI backend.

declare global {
  interface ImportMetaEnv {
    readonly VITE_API_URL?: string;
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

export interface RouteRequest {
  start: string;
  goal: string;
  algorithm?: 'astar' | 'dijkstra';
}

export interface RouteResponse {
  path: string[];
  geometry: [number, number][];
  directions: {
    step: number;
    instruction: string;
    node: string;
    distance_m: number;
  }[];
  safety_score: number;
  metadata: {
    total_distance_m: number;
    estimated_time_s: number;
    p_safe: number;
    num_turns: number;
    mean_edge_risk?: number;
    max_edge_risk?: number;
  };
}

export interface CompareResponse {
  baseline: RouteResponse;
  risk_aware: RouteResponse;
  comparison: {
    distance_increase_pct: number | null;
    safety_improvement: number;
  };
}

const BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

export async function getRoute(params: RouteRequest): Promise<RouteResponse> {
  const { start, goal, algorithm = 'astar' } = params;
  const res = await fetch(
    `${BASE_URL}/route?start=${start}&goal=${goal}&algorithm=${algorithm}`
  );
  if (!res.ok) throw new Error(`Failed to fetch route: ${res.statusText}`);
  return res.json();
}

export async function compareRoutes(start: string, goal: string): Promise<CompareResponse> {
  const res = await fetch(`${BASE_URL}/compare?start=${start}&goal=${goal}`);
  if (!res.ok) throw new Error(`Failed to compare routes: ${res.statusText}`);
  return res.json();
}

export async function getConfig() {
  const res = await fetch(`${BASE_URL}/config`);
  if (!res.ok) throw new Error(`Failed to load config`);
  return res.json();
}

export async function healthCheck() {
  const res = await fetch(`${BASE_URL}/health`);
  return res.json();
}
