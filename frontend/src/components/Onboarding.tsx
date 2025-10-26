import { useState } from "react";
import { motion } from "motion/react";
import { MapPin, Info, Navigation } from "lucide-react";

interface OnboardingProps {
  onComplete: () => void;
}

export function Onboarding({ onComplete }: OnboardingProps) {
  const [showPermissionModal, setShowPermissionModal] = useState(false);
  const [showInfo, setShowInfo] = useState(false);

  const handleStart = () => {
    setShowPermissionModal(true);
  };

  const handleAllowLocation = () => {
    onComplete();
  };

  const handleManualLocation = () => {
    onComplete();
  };

  return (
    <div className="min-h-screen bg-[#FDF8F0] flex flex-col items-center justify-center p-6 relative">
      {/* Main Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        {/* App Name with Logo */}
        <div className="mb-12 text-center">
          {/* Navigation-inspired Logo */}
          <div className="flex justify-center mb-6">
            <div className="border-3 border-black p-3">
              <svg width="50" height="50" viewBox="0 0 50 50" fill="none">
                <defs>
                  <linearGradient id="routeGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style={{ stopColor: "#AE1E2A", stopOpacity: 1 }} />
                    <stop offset="50%" style={{ stopColor: "#04771B", stopOpacity: 1 }} />
                    <stop offset="100%" style={{ stopColor: "#000000", stopOpacity: 1 }} />
                  </linearGradient>
                </defs>
                {/* Route path */}
                <path d="M10 40 L20 25 L30 20 L40 10" stroke="url(#routeGradient)" strokeWidth="3" />
                {/* Location markers */}
                <circle cx="10" cy="40" r="4" fill="url(#routeGradient)" stroke="#000" strokeWidth="1.5" />
                <circle cx="40" cy="10" r="4" fill="url(#routeGradient)" stroke="#000" strokeWidth="1.5" />
                {/* Directional arrow */}
                <path d="M30 20 L40 10 L35 18" stroke="url(#routeGradient)" strokeWidth="2.5" fill="none" />
              </svg>
            </div>
          </div>
          <h1 className="text-5xl mb-4 tracking-tight" style={{ fontWeight: 900 }}>SAFENAV</h1>
          <div className="border-2 border-black p-6 bg-[#FDF8F0]">
            <p className="text-base">Navigate safely during protests — no account required.</p>
          </div>
        </div>

        {/* Primary Button */}
        <motion.button
          whileTap={{ scale: 0.98 }}
          onClick={handleStart}
          className="w-full border-3 border-black bg-black text-[#FDF8F0] p-4 mb-4 hover:border-[#04771B] transition-colors text-center"
          style={{ fontWeight: 700 }}
        >
          START
        </motion.button>

        {/* Info Button */}
        <button
          onClick={() => setShowInfo(!showInfo)}
          className="w-full border-2 border-black bg-[#FDF8F0] p-3 mb-4 flex items-center justify-center gap-2 hover:bg-[#E8E3D8] transition-colors"
        >
          <Info size={18} />
          <span className="text-sm">Why we need your location</span>
        </button>

        {/* Info Expansion */}
        {showInfo && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            className="border-2 border-black p-4 mb-4 bg-[#E8E3D8]"
          >
            <p className="text-sm">
              Location data is used to show nearby hazards and compute safe routes. All data stays on your device. No tracking, no servers.
            </p>
          </motion.div>
        )}

        {/* Manual Location Option */}
        <button
          onClick={handleManualLocation}
          className="w-full text-sm underline hover:no-underline"
        >
          Use manual location
        </button>
      </motion.div>

      {/* Location Permission Modal */}
      {showPermissionModal && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 bg-black/60 flex items-center justify-center p-6 z-50"
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            className="w-full max-w-sm border-3 border-black bg-[#FDF8F0] p-6"
          >
            <div className="flex items-center justify-center mb-6">
              <div className="border-2 border-black p-4 bg-[#04771B]">
                <MapPin size={32} className="text-black" />
              </div>
            </div>
            
            <h2 className="text-xl mb-3 text-center">Location Access</h2>
            <p className="text-sm mb-6 text-center">
              SafeNav needs your location to show hazards and routes near you.
            </p>

            <button
              onClick={handleAllowLocation}
              className="w-full border-3 border-black bg-[rgb(4,119,27)] text-black p-3 mb-3 hover:border-[#000] transition-colors"
            >
              ALLOW LOCATION
            </button>

            <button
              onClick={handleManualLocation}
              className="w-full border-2 border-black bg-[#FDF8F0] text-black p-3 hover:bg-[#E8E3D8] transition-colors"
            >
              USE MANUAL LOCATION
            </button>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
}