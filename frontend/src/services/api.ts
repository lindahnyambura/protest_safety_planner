const API_BASE_URL = import.meta.env.VITE_API_URL;

export const apiService = {
  async submitReport(report: any) {
    const response = await fetch(`${API_BASE_URL}/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(report),
    });
    return response.json();
  },

  async getNearestNode(lat: number, lng: number) {
    const response = await fetch(`${API_BASE_URL}/nearest-node?lat=${lat}&lng=${lng}`);
    return response.json();
  },

  async getRoute(start: string, goal: string, riskPreference: number) {
    const response = await fetch(
      `${API_BASE_URL}/route?start=${start}&goal=${goal}&algorithm=astar&lambda_risk=${riskPreference}`
    );
    return response.json();
  },
};