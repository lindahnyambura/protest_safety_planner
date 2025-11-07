// src/services/api.ts
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface ReportData {
  type: string;
  lat: number;
  lng: number;
  confidence: number;
  notes?: string;
  timestamp?: string;
}

export interface ReportResponse {
  status: string;
  report_id: string;
  node_id: string;
  snapped_distance_m: number;
  expires_in_seconds: number;
}

class ApiService {
  async submitReport(report: ReportData): Promise<ReportResponse> {
    const response = await fetch(`${API_BASE_URL}/report`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(report),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to submit report: ${response.status} - ${errorText}`);
    }

    return response.json();
  }

  async getNearestNode(lat: number, lng: number): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/nearest-node?lat=${lat}&lng=${lng}`);
    if (!response.ok) throw new Error('Failed to find nearest node');
    return response.json();
  }

  async getHealth(): Promise<any> {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error('Backend health check failed');
    return response.json();
  }
}

export const apiService = new ApiService();