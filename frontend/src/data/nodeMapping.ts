// Map common Nairobi CBD landmarks to your actual node IDs
export const NODE_MAPPINGS: Record<string, string> = {
  // You'll need to replace these with actual node IDs from your graph
  "bus station": "node_123", 
  "national archives": "node_456",
  "railways": "node_789",
  "uhuru park": "node_101",
  "central station": "node_202",
  "kicc": "node_303",
  "nairobi hospital": "node_404",
};

// Common starting points
export const DEFAULT_START_NODE = "node_42"; // Replace with actual node ID 