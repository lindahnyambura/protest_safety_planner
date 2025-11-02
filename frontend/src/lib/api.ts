// API client for backend communication
const API_BASE = 'http://localhost:8000';

export interface RouteRequest {
  start: string;
  goal: string;
  algorithm?: string;
}

export interface RouteResponse {
  path: string[];
  geometry: [number, number][];
  directions: Array<{
    step: number;
    instruction: string;
    node: string;
    distance_m: number;
  }>;
  safety_score: number;
  metadata: any;
  algorithm: string;
}

export interface RiskMapResponse {
  type: string;
  features: Array<{
    type: string;
    geometry: {
      type: string;
      coordinates: [number, number];
    };
    properties: {
      risk: number;
      cell_id: string;
    };
  }>;
}

export interface Alert {
  id: string;
  type: string;
  location: string;
  time: string;
  severity: "low" | "medium" | "high";
}

class ApiClient {
  async getHealth(): Promise<{ status: string }> {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
  }

  async getRoute(start: string, goal: string, algorithm = 'astar'): Promise<RouteResponse> {
    const response = await fetch(
      `${API_BASE}/route?start=${encodeURIComponent(start)}&goal=${encodeURIComponent(goal)}&algorithm=${algorithm}`
    );
    if (!response.ok) {
      throw new Error(`Route request failed: ${response.statusText}`);
    }
    return response.json();
  }

  async getRiskMap(): Promise<RiskMapResponse> {
    const response = await fetch(`${API_BASE}/riskmap`);
    if (!response.ok) {
      throw new Error(`Risk map request failed: ${response.statusText}`);
    }
    return response.json();
  }

  async submitReport(reportData: {
    type: string;
    location: string;
    timestamp: string;
  }): Promise<{ status: string; id: string }> {
    const response = await fetch(`${API_BASE}/report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(reportData),
    });
    return response.json();
  }

  async getAlerts(): Promise<Alert[]> {
    const response = await fetch(`${API_BASE}/alerts`);
    return response.json();
  }
}

export const apiClient = new ApiClient();