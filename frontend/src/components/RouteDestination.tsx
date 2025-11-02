// frontend/src/components/RouteDestination.tsx
import { useState } from "react";
import { motion } from "motion/react";
import { ArrowLeft, Search, MapPin, Navigation } from "lucide-react";
import { apiClient } from "../lib/api";
import { toast } from "sonner";

interface RouteDestinationProps {
  onNavigate: (screen: string, data?: any) => void;
  onBack: () => void;
}

// Map preset destinations to actual node IDs from your graph
const PRESET_DESTINATIONS = [
  { id: "1", name: "Nearest Exit", icon: Navigation, nodeId: "node_123" }, // You'll need real node IDs
  { id: "2", name: "Bus Station", icon: MapPin, nodeId: "10873342299" },
  { id: "3", name: "National Archives", icon: MapPin, nodeId: "12361156623" },
  { id: "4", name: "Railways", icon: MapPin, nodeId: "8584796189" },
];

export function RouteDestination({ onNavigate, onBack }: RouteDestinationProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [riskBalance, setRiskBalance] = useState(50);
  const [isComputing, setIsComputing] = useState(false);
  const [selectedDestination, setSelectedDestination] = useState<string | null>(null);

  const handleComputeRoute = async () => {
    if (!selectedDestination && !searchQuery) {
      toast.error("Please select a destination");
      return;
    }

    setIsComputing(true);
    
    try {
      // For demo, we'll use mock start node - in production, get from user location
      const startNode = await getCurrentLocationNode(); // You'll implement this
      const goalNode = selectedDestination || await searchForNode(searchQuery);
      
      if (!goalNode) {
        toast.error("Destination not found. Please try another location.");
        return;
      }

      // Convert risk balance to algorithm preference
      const algorithm = riskBalance < 33 ? "astar" : "dijkstra"; // Safest vs balanced
      
      console.log(`Computing route from ${startNode} to ${goalNode} with ${algorithm}`);
      
      const route = await apiClient.getRoute(startNode, goalNode, algorithm);
      
      // Navigate to route details with the computed route
      onNavigate("route-details", { route });
      
    } catch (error) {
      console.error("Failed to compute route:", error);
      toast.error("Failed to compute route. Please try again.");
      // Fallback to mock data for demo
      onNavigate("route-details", { 
        route: getMockRouteData() 
      });
    } finally {
      setIsComputing(false);
    }
  };

  // Mock function to get current location node - you'll replace this
  const getCurrentLocationNode = async (): Promise<string> => {
    // In production, you'd:
    // 1. Get user's current lat/lng
    // 2. Find nearest node in your graph
    // 3. Return node ID
    
    // For demo, return a fixed node ID from Nairobi CBD
    return "node_42"; // Replace with actual node ID from your graph
  };

  // Mock function to search for node - you'll replace this
  const searchForNode = async (query: string): Promise<string | null> => {
    // In production, you'd:
    // 1. Search your graph for nodes matching the query
    // 2. Return the best match node ID
    
    // For demo, map common names to node IDs
    const searchMap: Record<string, string> = {
      "bus station": "10873342299",
      "national archives": "12361156623", 
      "railways": "8584796189",
      "uhuru park": "12343642875",
      "times tower": "10701041875",
      "teleposta towers": "9859577513",
      "kicc": "13134429074",
    };
    
    return searchMap[query.toLowerCase()] || null;
  };

  const handlePresetSelect = (dest: typeof PRESET_DESTINATIONS[0]) => {
    setSearchQuery(dest.name);
    setSelectedDestination(dest.nodeId);
  };

  const getMockRouteData = () => ({
    path: ["node_42", "node_43", "node_44", "node_45"],
    geometry: [
      [36.8172, -1.2833],
      [36.8175, -1.2830],
      [36.8178, -1.2827],
      [36.8180, -1.2825]
    ],
    directions: [
      { step: 0, instruction: "Start at your location", node: "node_42", distance_m: 0 },
      { step: 1, instruction: "Head west on Moi Ave", node: "node_43", distance_m: 120 },
      { step: 2, instruction: "Turn right onto Haile Selassie Ave", node: "node_44", distance_m: 450 },
      { step: 3, instruction: "Arrive at destination", node: "node_45", distance_m: 200 }
    ],
    safety_score: 0.85,
    metadata: {
      total_distance_m: 770,
      total_risk: 0.15,
      algorithm: "astar"
    },
    algorithm: "astar"
  });

  return (
    <div className="min-h-screen bg-[#FDF8F0] flex flex-col">
      {/* Header */}
      <div className="border-b-2 border-black bg-[#FDF8F0] p-4 flex items-center gap-3">
        <button
          onClick={onBack}
          className="border-2 border-black p-2 hover:bg-[#E8E3D8] transition-colors"
        >
          <ArrowLeft size={20} />
        </button>
        <h1 className="text-lg">FIND SAFE ROUTE</h1>
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
        {/* Search Input */}
        <div className="mb-6">
          <label className="block mb-3 text-sm" style={{ fontWeight: 700 }}>DESTINATION</label>
          <div className="relative">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setSelectedDestination(null); // Clear preset selection when typing
              }}
              placeholder="Search location..."
              className="w-full border-2 border-black p-3 pl-12 bg-[#FDF8F0] focus:border-[#04771B] outline-none transition-colors"
            />
            <Search className="absolute left-3 top-1/2 -translate-y-1/2" size={20} />
          </div>
        </div>

        {/* Preset Destinations */}
        <div className="mb-6">
          <label className="block mb-3 text-sm" style={{ fontWeight: 700 }}>QUICK DESTINATIONS</label>
          <div className="grid grid-cols-2 gap-3">
            {PRESET_DESTINATIONS.map((dest) => {
              const Icon = dest.icon;
              const isSelected = selectedDestination === dest.nodeId;
              
              return (
                <motion.button
                  key={dest.id}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => handlePresetSelect(dest)}
                  className={`border-2 border-black p-4 flex items-center gap-2 transition-colors ${
                    isSelected 
                      ? 'bg-[#04771B] text-white' 
                      : 'bg-[#FDF8F0] hover:bg-[#E8E3D8]'
                  }`}
                >
                  <Icon size={18} />
                  <span className="text-sm">{dest.name}</span>
                </motion.button>
              );
            })}
          </div>
        </div>

        {/* Risk vs Distance Slider */}
        <div className="mb-8">
          <label className="block mb-3 text-sm" style={{ fontWeight: 700 }}>ROUTE PREFERENCE</label>
          <div className="border-2 border-black p-4 bg-[#E8E3D8]">
            <div className="flex justify-between mb-3 text-xs">
              <span>SAFEST</span>
              <span>BALANCED</span>
              <span>FASTEST</span>
            </div>
            <div className="relative">
              <input
                type="range"
                min="0"
                max="100"
                value={riskBalance}
                onChange={(e) => setRiskBalance(Number(e.target.value))}
                className="w-full h-3 appearance-none bg-[#FDF8F0] border-2 border-black cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #04771B 0%, #04771B ${riskBalance}%, #FDF8F0 ${riskBalance}%, #FDF8F0 100%)`,
                }}
              />
              <style>{`
                input[type="range"]::-webkit-slider-thumb {
                  appearance: none;
                  width: 24px;
                  height: 24px;
                  background: #000;
                  border: 2px solid #000;
                  cursor: pointer;
                }
                input[type="range"]::-moz-range-thumb {
                  width: 24px;
                  height: 24px;
                  background: #000;
                  border: 2px solid #000;
                  cursor: pointer;
                }
              `}</style>
            </div>
            <p className="text-xs mt-3 text-center">
              {riskBalance < 33 ? "Prioritizing safety over speed" :
               riskBalance > 66 ? "Prioritizing speed over safety" :
               "Balanced route"}
            </p>
          </div>
        </div>

        {/* Info Box */}
        <div className="border-2 border-black p-4 bg-[#FDF8F0] mb-6">
          <p className="text-xs">
            <strong>How it works:</strong> Routes avoid reported hazards and high-risk zones. 
            Adjust the slider to balance safety vs. travel time.
          </p>
        </div>
      </div>

      {/* Bottom Action Button */}
      <div className="border-t-2 border-black p-4 bg-[#FDF8F0]">
        <motion.button
          whileTap={{ scale: 0.98 }}
          onClick={handleComputeRoute}
          disabled={(!selectedDestination && !searchQuery) || isComputing}
          className="w-full border-3 border-black bg-[#04771B] text-black p-4 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#19647E] transition-colors text-center"
          style={{ fontWeight: 600 }}
        >
          {isComputing ? 'COMPUTING ROUTE...' : 'COMPUTE SAFE ROUTE'}
        </motion.button>
      </div>
    </div>
  );
}