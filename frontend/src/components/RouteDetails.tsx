import { useState } from "react";
import { motion } from "motion/react";
import { ArrowLeft, Navigation, Share2, ChevronDown, ChevronUp, Clock, TrendingUp } from "lucide-react";
import { MapView } from "./MapView";
import { toast } from "sonner";

interface RouteDetailsProps {
  onNavigate: (screen: string) => void;
  onBack: () => void;
}

export function RouteDetails({ onNavigate, onBack }: RouteDetailsProps) {
  const [stepsExpanded, setStepsExpanded] = useState(false);

  const routeSteps = [
    { id: "1", instruction: "Head west on Moi Ave", distance: "120 m", risk: "low" },
    { id: "2", instruction: "Turn right onto Haile Selassie Ave", distance: "450 m", risk: "low" },
    { id: "3", instruction: "Continue straight (avoid Tom Mboya St)", distance: "200 m", risk: "medium" },
    { id: "4", instruction: "Turn left onto Kenyatta Ave", distance: "300 m", risk: "low" },
    { id: "5", instruction: "Arrive at Uhuru Park", distance: "50 m", risk: "low" },
  ];

  const handleShare = () => {
    toast("Route copied as text", {
      description: "You can now share it safely",
    });
  };

  const handleStartGuidance = () => {
    onNavigate("live-guidance");
  };

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
        <MapView showUserLocation showHazards showRoute />
      </div>

      {/* Route Stats Card */}
      <div className="border-b-2 border-black bg-[#04771B] p-4">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-xs opacity-80 mb-1">SAFETY SCORE</p>
            <div className="flex items-center justify-center gap-1">
              <TrendingUp size={16} />
              <span className="text-xl">8.5</span>
            </div>
          </div>
          <div>
            <p className="text-xs opacity-80 mb-1">ETA</p>
            <div className="flex items-center justify-center gap-1">
              <Clock size={16} />
              <span className="text-xl">12 min</span>
            </div>
          </div>
          <div>
            <p className="text-xs opacity-80 mb-1">DISTANCE</p>
            <span className="text-xl">1.1 km</span>
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
            {routeSteps.map((step, index) => (
              <div key={step.id} className="p-4 flex gap-3">
                <div className="border-2 border-black w-8 h-8 flex items-center justify-center shrink-0 bg-[#FDF8F0]">
                  {index + 1}
                </div>
                <div className="flex-1">
                  <p className="text-sm mb-1">{step.instruction}</p>
                  <div className="flex items-center gap-3 text-xs opacity-70">
                    <span>{step.distance}</span>
                    <span 
                      className="border border-black px-2 py-0.5"
                      style={{ 
                        backgroundColor: step.risk === "low" ? "#04771B" : 
                                       step.risk === "medium" ? "#19647E" : "#AE1E2A" 
                      }}
                    >
                      {step.risk.toUpperCase()} RISK
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
                <strong>Route Summary:</strong> This route avoids 2 reported hazards and 
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