import { useState } from "react";
import { motion } from "motion/react";
import { ArrowLeft, Search, MapPin, Navigation } from "lucide-react";

interface RouteDestinationProps {
  onNavigate: (screen: string, data?: any) => void;
  onBack: () => void;
}

export function RouteDestination({ onNavigate, onBack }: RouteDestinationProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [riskBalance, setRiskBalance] = useState(50);

  const presetDestinations = [
    { id: "1", name: "Nearest Exit", icon: Navigation },
    { id: "2", name: "Uhuru Park", icon: MapPin },
    { id: "3", name: "Central Station", icon: MapPin },
    { id: "4", name: "Medical Center", icon: MapPin },
  ];

  const handleComputeRoute = () => {
    onNavigate("route-details");
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
              onChange={(e) => setSearchQuery(e.target.value)}
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
            {presetDestinations.map((dest) => {
              const Icon = dest.icon;
              return (
                <motion.button
                  key={dest.id}
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setSearchQuery(dest.name)}
                  className="border-2 border-black p-4 flex items-center gap-2 hover:bg-[#E8E3D8] transition-colors"
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
          disabled={!searchQuery}
          className="w-full border-3 border-black bg-[#04771B] text-black p-4 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-[#19647E] transition-colors text-center"
          style={{ fontWeight: 600 }}
        >
          COMPUTE SAFE ROUTE
        </motion.button>
      </div>
    </div>
  );
}