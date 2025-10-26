import { useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { ArrowLeft, Navigation, AlertTriangle, X } from "lucide-react";
import { MapView } from "./MapView";

interface LiveGuidanceProps {
  onNavigate: (screen: string) => void;
  onBack: () => void;
}

export function LiveGuidance({ onNavigate, onBack }: LiveGuidanceProps) {
  const [showRerouteAlert, setShowRerouteAlert] = useState(true);
  const [currentStep, setCurrentStep] = useState(1);

  const totalSteps = 5;
  const progress = (currentStep / totalSteps) * 100;

  const handleReroute = () => {
    setShowRerouteAlert(false);
    onNavigate("route-destination");
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
        <h1 className="text-lg flex-1">LIVE GUIDANCE</h1>
        <div className="border-2 border-black p-2 bg-[#04771B]">
          <Navigation size={20} />
        </div>
      </div>

      {/* Map View */}
      <div className="flex-1 relative">
        <MapView showUserLocation showHazards showRoute />

        {/* Reroute Alert Modal */}
        <AnimatePresence>
          {showRerouteAlert && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 bg-black/60 flex items-center justify-center p-4 z-20"
            >
              <motion.div
                initial={{ scale: 0.9, y: 20 }}
                animate={{ scale: 1, y: 0 }}
                exit={{ scale: 0.9, y: 20 }}
                className="w-full max-w-sm border-3 border-black bg-[#AE1E2A] p-6"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="border-2 border-black p-2 bg-[#FDF8F0]">
                    <AlertTriangle size={24} />
                  </div>
                  <button
                    onClick={() => setShowRerouteAlert(false)}
                    className="border-2 border-black p-1 bg-[#FDF8F0] hover:bg-[#E8E3D8] transition-colors"
                  >
                    <X size={16} />
                  </button>
                </div>

                <h3 className="text-lg mb-2">NEW HAZARD DETECTED</h3>
                <p className="text-sm mb-4">
                  Tear gas reported ahead on Tom Mboya St. We recommend a safer route.
                </p>

                <div className="space-y-2">
                  <button
                    onClick={handleReroute}
                    className="w-full border-3 border-black bg-black text-[#FDF8F0] p-3 hover:bg-[#6B7F59] hover:text-black transition-colors"
                  >
                    FIND SAFER ROUTE
                  </button>
                  <button
                    onClick={() => setShowRerouteAlert(false)}
                    className="w-full border-2 border-black bg-[#FDF8F0] text-black p-3 hover:bg-[#E8E3D8] transition-colors"
                  >
                    CONTINUE ANYWAY
                  </button>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Instruction Banner */}
      <div className="border-t-3 border-black bg-[#04771B] p-6">
        <motion.div
          key={currentStep}
          initial={{ x: 20, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="text-center"
        >
          <p className="text-xs opacity-80 mb-2">NEXT INSTRUCTION</p>
          <h2 className="text-2xl mb-2">Head west on Moi Ave</h2>
          <p className="text-xl">120 m</p>
        </motion.div>
      </div>

      {/* Progress and Safety Indicator */}
      <div className="border-t-2 border-black bg-[#FDF8F0] p-4">
        <div className="mb-3">
          <div className="flex justify-between text-xs mb-2">
            <span>PROGRESS</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="h-3 border-2 border-black bg-[#E8E3D8]">
            <motion.div
              className="h-full bg-[#04771B]"
              initial={{ width: 0 }}
              animate={{ width: `${progress}%` }}
              transition={{ duration: 0.5 }}
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 text-center text-xs">
          <div className="border-2 border-black p-2 bg-[#E8E3D8]">
            <p className="opacity-70 mb-1">ETA</p>
            <p>8 min</p>
          </div>
          <div className="border-2 border-black p-2 bg-[#04771B]">
            <p className="opacity-70 mb-1">SAFETY</p>
            <p>HIGH</p>
          </div>
        </div>
      </div>
    </div>
  );
}