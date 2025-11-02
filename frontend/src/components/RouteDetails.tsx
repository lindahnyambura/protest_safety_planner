// frontend/src/components/RouteDetails.tsx
import { useState, useEffect } from "react";
import { motion } from "motion/react";
import { ArrowLeft, Navigation, Share2, ChevronDown, ChevronUp, Clock, TrendingUp } from "lucide-react";
import { MapView } from "./MapView";
import { toast } from "sonner";

interface RouteDetailsProps {
  onNavigate: (screen: string, data?: any) => void;
  onBack: () => void;
  routeData?: any; // The route data passed from RouteDestination
}

export function RouteDetails({ onNavigate, onBack, routeData }: RouteDetailsProps) {
  const [stepsExpanded, setStepsExpanded] = useState(false);
  const [route, setRoute] = useState<any>(null);

  useEffect(() => {
    if (routeData?.route) {
      setRoute(routeData.route);
    } else {
      // Fallback to mock data if no route provided
      setRoute(getMockRouteData());
    }
  }, [routeData]);

  const handleShare = () => {
    if (route) {
      const routeText = route.directions
        .map((step: any) => `${step.instruction} (${step.distance_m}m)`)
        .join('\n');
      
      navigator.clipboard.writeText(routeText);
      toast("Route copied as text", {
        description: "You can now share it safely",
      });
    }
  };

  const handleStartGuidance = () => {
    if (route) {
      onNavigate("live-guidance", { route });
    }
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
      { step: 0, instruction: "Start at your location", node: "node_42", distance_m: 0, risk: "low" },
      { step: 1, instruction: "Head west on Moi Ave", node: "node_43", distance_m: 120, risk: "low" },
      { step: 2, instruction: "Turn right onto Haile Selassie Ave", node: "node_44", distance_m: 450, risk: "low" },
      { step: 3, instruction: "Continue straight (avoid Tom Mboya St)", node: "node_45", distance_m: 200, risk: "medium" },
      { step: 4, instruction: "Arrive at destination", node: "node_46", distance_m: 50, risk: "low" }
    ],
    safety_score: 0.85,
    metadata: {
      total_distance_m: 820,
      total_risk: 0.15,
      algorithm: "astar",
      estimated_time_min: 12
    },
    algorithm: "astar"
  });

  if (!route) {
    return (
      <div className="min-h-screen bg-[#FDF8F0] flex items-center justify-center">
        <p>Loading route...</p>
      </div>
    );
  }

  const totalDistanceKm = (route.metadata.total_distance_m / 1000).toFixed(1);
  const safetyScore = Math.round(route.safety_score * 10);
  const estimatedTime = route.metadata.estimated_time_min || 12;

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
        <h1 className="text-lg flex-1">ROUTE DETAILS</h1>
        <button
          onClick={handleShare}
          className="border-2 border-black p-2 hover:bg-[#E8E3D8] transition-colors"
        >
          <Share2 size={20} />
        </button>
      </div>

      {/* Map View */}
      <div className="h-64 border-b-2 border-black">
        <MapView 
          showUserLocation 
          showHazards 
          showRoute 
          routeGeometry={route.geometry}
        />
      </div>

      {/* Route Stats Card */}
      <div className="border-b-2 border-black bg-[#04771B] p-4">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-xs opacity-80 mb-1">SAFETY SCORE</p>
            <div className="flex items-center justify-center gap-1">
              <TrendingUp size={16} />
              <span className="text-xl">{safetyScore}/10</span>
            </div>
          </div>
          <div>
            <p className="text-xs opacity-80 mb-1">ETA</p>
            <div className="flex items-center justify-center gap-1">
              <Clock size={16} />
              <span className="text-xl">{estimatedTime} min</span>
            </div>
          </div>
          <div>
            <p className="text-xs opacity-80 mb-1">DISTANCE</p>
            <span className="text-xl">{totalDistanceKm} km</span>
          </div>
        </div>
      </div>

      {/* Steps Section */}
      <div className="flex-1 overflow-y-auto">
        <button
          onClick={() => setStepsExpanded(!stepsExpanded)}
          className="w-full p-4 flex items-center justify-between border-b-2 border-black hover:bg-[#E8E3D8] transition-colors"
        >
          <span className="text-sm">STEP-BY-STEP DIRECTIONS</span>
          {stepsExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>

        {stepsExpanded && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="divide-y-2 divide-black"
          >
            {route.directions.map((step: any, index: number) => (
              <div key={step.step} className="p-4 flex gap-3">
                <div className="border-2 border-black w-8 h-8 flex items-center justify-center shrink-0 bg-[#FDF8F0]">
                  {index + 1}
                </div>
                <div className="flex-1">
                  <p className="text-sm mb-1">{step.instruction}</p>
                  <div className="flex items-center gap-3 text-xs opacity-70">
                    <span>{step.distance_m}m</span>
                    <span 
                      className="border border-black px-2 py-0.5"
                      style={{ 
                        backgroundColor: step.risk === "low" ? "#04771B" : 
                                       step.risk === "medium" ? "#19647E" : "#AE1E2A" 
                      }}
                    >
                      {step.risk?.toUpperCase() || "LOW"} RISK
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </motion.div>
        )}

        {/* Additional Info */}
        {!stepsExpanded && (
          <div className="p-4 border-b-2 border-black">
            <div className="border-2 border-black p-3 bg-[#E8E3D8]">
              <p className="text-xs">
                <strong>Route Summary:</strong> This route avoids reported hazards and 
                prioritizes main roads with good visibility. Tap to see full directions.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Actions */}
      <div className="border-t-2 border-black p-4 bg-[#FDF8F0] space-y-3">
        <motion.button
          whileTap={{ scale: 0.98 }}
          onClick={handleStartGuidance}
          className="w-full border-3 border-black bg-black text-[#FDF8F0] p-4 flex items-center justify-center gap-2 hover:bg-[#6B7F59] hover:text-black transition-colors"
        >
          <Navigation size={20} />
          START GUIDANCE
        </motion.button>
        
        <p className="text-xs text-center opacity-60">
          Turn-by-turn navigation with real-time alerts
        </p>
      </div>
    </div>
  );
}