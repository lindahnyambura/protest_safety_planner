import { useState } from "react";
import { motion } from "motion/react";
import { AlertCircle, MapPin, Route, Menu, Bell, Settings as SettingsIcon, ChevronUp, ChevronDown } from "lucide-react";
import { MapView } from "./MapView";

interface Event {
  id: string;
  type: string;
  location: string;
  time: string;
  severity: "low" | "medium" | "high";
}

interface HomeMapProps {
  onNavigate: (screen: string, data?: any) => void;
}

export function HomeMap({ onNavigate }: HomeMapProps) {
  const [eventsExpanded, setEventsExpanded] = useState(false);
  const [riskLevel, setRiskLevel] = useState<"low" | "moderate" | "high">("low");

  const mockEvents: Event[] = [
    { id: "1", type: "Tear gas reported", location: "Tom Mboya St", time: "2 min ago", severity: "high" },
    { id: "2", type: "Heavy crowd", location: "University Way", time: "8 min ago", severity: "medium" },
    { id: "3", type: "Police presence", location: "Kenyatta Ave", time: "15 min ago", severity: "medium" },
  ];

  const getRiskColor = () => {
    switch (riskLevel) {
      case "low": return "#04771B";
      case "moderate": return "#19647E";
      case "high": return "#AE1E2A";
    }
  };

  const getRiskText = () => {
    switch (riskLevel) {
      case "low": return "You are in a LOW RISK area";
      case "moderate": return "You are in a MODERATE RISK area";
      case "high": return "You are in a HIGH RISK area";
    }
  };

  return (
    <div className="min-h-screen bg-[#FDF8F0] flex flex-col">
      {/* Simplified Top Bar */}
      <div className="border-b-2 border-black bg-[#FDF8F0] p-3 flex items-center justify-between">
        <button 
          onClick={() => onNavigate("settings")}
          className="border-2 border-black p-2 hover:bg-[#E8E3D8] transition-colors"
        >
          <Menu size={18} />
        </button>
        <h1 className="text-base" style={{ fontWeight: 600 }}>SAFENAV</h1>
        <button 
          onClick={() => onNavigate("alerts")}
          className="border-2 border-black p-2 hover:bg-[#E8E3D8] transition-colors"
        >
          <Bell size={18} />
        </button>
      </div>

      {/* Map Container */}
      <div className="flex-1 relative">
        <MapView showUserLocation showHazards />

        {/* Risk Status Card - moved lower to avoid nav bar */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="absolute top-16 left-4 right-4 border-3 border-black p-3 max-w-[calc(100%-6rem)]"
          style={{ backgroundColor: getRiskColor() }}
        >
          <div className="flex items-center gap-2">
            <div className="border-2 border-black p-1.5 bg-[#FDF8F0] flex-shrink-0">
              <AlertCircle size={16} />
            </div>
            <div className="min-w-0">
              <p className="text-xs opacity-80">STATUS</p>
              <p className="text-xs truncate" style={{ fontWeight: 600 }}>{getRiskText()}</p>
            </div>
          </div>
        </motion.div>

        {/* Current Location Label - positioned to not overlap */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute top-4 right-4 border-2 border-black bg-[#FDF8F0] px-3 py-1.5 text-xs max-w-[40%]"
        >
          <div className="flex items-center gap-1.5">
            <MapPin size={12} />
            <span className="truncate">Uhuru Park</span>
          </div>
        </motion.div>

        {/* Floating Action Buttons */}
        <div className="absolute bottom-32 right-4 flex flex-col gap-3">
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={() => onNavigate("quick-report")}
            className="border-3 border-black bg-[#AE1E2A] text-black px-4 py-3 hover:border-[#000] transition-colors"
          >
            <AlertCircle size={18} className="inline mr-2" />
            REPORT
          </motion.button>
          <motion.button
            whileTap={{ scale: 0.95 }}
            onClick={() => onNavigate("route-destination")}
            className="border-3 border-black bg-[#FDF8F0] text-black px-4 py-3 hover:bg-[#E8E3D8] transition-colors"
          >
            <Route size={18} className="inline mr-2" />
            SAFE ROUTE
          </motion.button>
        </div>
      </div>

      {/* Bottom Events Card */}
      <motion.div
        animate={{ height: eventsExpanded ? "auto" : "120px" }}
        className="border-t-3 border-black bg-[#FDF8F0] overflow-hidden"
      >
        <button
          onClick={() => setEventsExpanded(!eventsExpanded)}
          className="w-full p-4 flex items-center justify-between border-b-2 border-black hover:bg-[#E8E3D8] transition-colors"
        >
          <span className="text-sm">RECENT EVENTS</span>
          {eventsExpanded ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
        </button>

        <div className="divide-y-2 divide-black">
          {mockEvents.map((event, index) => (
            <motion.button
              key={event.id}
              onClick={() => onNavigate("alerts")}
              className="w-full p-4 flex items-center gap-3 hover:bg-[#E8E3D8] transition-colors text-left"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <div 
                className="border-2 border-black p-2"
                style={{ 
                  backgroundColor: event.severity === "high" ? "#AE1E2A" : 
                                 event.severity === "medium" ? "#19647E" : "#04771B" 
                }}
              >
                <AlertCircle size={16} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm truncate">{event.type}</p>
                <p className="text-xs opacity-70 truncate">{event.location} · {event.time}</p>
              </div>
            </motion.button>
          ))}
        </div>
      </motion.div>
    </div>
  );
}