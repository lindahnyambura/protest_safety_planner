import { useState } from "react";
import { motion } from "motion/react";
import { ArrowLeft, Wind, AlertTriangle, Users, Filter } from "lucide-react";

interface Alert {
  id: string;
  type: "gas" | "police" | "crowd";
  title: string;
  location: string;
  time: string;
  description: string;
  confidence: "low" | "medium" | "high";
}

interface AlertsFeedProps {
  onBack: () => void;
  onSelectAlert?: (alert: Alert) => void;
}

export function AlertsFeed({ onBack, onSelectAlert }: AlertsFeedProps) {
  const [activeFilter, setActiveFilter] = useState<string | null>(null);

  const mockAlerts: Alert[] = [
    {
      id: "1",
      type: "gas",
      title: "Tear gas reported",
      location: "Tom Mboya St, near GPO",
      time: "2 min ago",
      description: "Heavy tear gas deployment reported by 12 users",
      confidence: "high",
    },
    {
      id: "2",
      type: "crowd",
      title: "Heavy crowd",
      location: "University Way",
      time: "8 min ago",
      description: "Large gathering, movement restricted",
      confidence: "high",
    },
    {
      id: "3",
      type: "police",
      title: "Police presence",
      location: "Kenyatta Ave intersection",
      time: "15 min ago",
      description: "Multiple units observed",
      confidence: "medium",
    },
    {
      id: "4",
      type: "gas",
      title: "Tear gas cleared",
      location: "Moi Ave",
      time: "22 min ago",
      description: "Area now passable",
      confidence: "medium",
    },
    {
      id: "5",
      type: "crowd",
      title: "Crowd dispersing",
      location: "City Square",
      time: "30 min ago",
      description: "Situation calming down",
      confidence: "low",
    },
  ];

  const filterOptions = [
    { id: "gas", label: "Gas", icon: Wind, color: "#AE1E2A" },
    { id: "police", label: "Police", icon: AlertTriangle, color: "#19647E" },
    { id: "crowd", label: "Crowd", icon: Users, color: "#E8E3D8" },
  ];

  const filteredAlerts = activeFilter
    ? mockAlerts.filter((alert) => alert.type === activeFilter)
    : mockAlerts;

  const getAlertColor = (type: string) => {
    switch (type) {
      case "gas": return "#AE1E2A";
      case "police": return "#19647E";
      case "crowd": return "#E8E3D8";
      default: return "#E8E3D8";
    }
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case "gas": return Wind;
      case "police": return AlertTriangle;
      case "crowd": return Users;
      default: return AlertTriangle;
    }
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
        <h1 className="text-lg flex-1">ALERTS & EVENTS</h1>
        <div className="border-2 border-black p-2 bg-[#E8E3D8]">
          <Filter size={20} />
        </div>
      </div>

      {/* Filter Bar */}
      <div className="border-b-2 border-black bg-[#FDF8F0] p-3 flex gap-2 overflow-x-auto">
        <button
          onClick={() => setActiveFilter(null)}
          className="border-2 border-black px-4 py-2 text-sm whitespace-nowrap hover:bg-[#E8E3D8] transition-colors"
          style={{
            backgroundColor: activeFilter === null ? "#E8E3D8" : "#FDF8F0",
            borderWidth: activeFilter === null ? "3px" : "2px",
          }}
        >
          ALL
        </button>
        {filterOptions.map((filter) => {
          const Icon = filter.icon;
          return (
            <button
              key={filter.id}
              onClick={() => setActiveFilter(filter.id)}
              className="border-2 border-black px-4 py-2 text-sm whitespace-nowrap flex items-center gap-2 hover:bg-[#E8E3D8] transition-colors"
              style={{
                backgroundColor: activeFilter === filter.id ? filter.color : "#FDF8F0",
                borderWidth: activeFilter === filter.id ? "3px" : "2px",
              }}
            >
              <Icon size={16} />
              {filter.label}
            </button>
          );
        })}
      </div>

      {/* Alerts List */}
      <div className="flex-1 overflow-y-auto">
        <div className="divide-y-2 divide-black">
          {filteredAlerts.map((alert, index) => {
            const Icon = getAlertIcon(alert.type);
            return (
              <motion.button
                key={alert.id}
                onClick={() => onSelectAlert?.(alert)}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className="w-full p-4 flex gap-3 hover:bg-[#E8E3D8] transition-colors text-left"
              >
                <div
                  className="border-2 border-black p-3 shrink-0"
                  style={{ backgroundColor: getAlertColor(alert.type) }}
                >
                  <Icon size={20} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <h3 className="text-sm">{alert.title}</h3>
                    <span className="text-xs opacity-60 whitespace-nowrap">{alert.time}</span>
                  </div>
                  <p className="text-xs opacity-70 mb-2">{alert.location}</p>
                  <p className="text-sm mb-2">{alert.description}</p>
                  <div
                    className="inline-block border border-black px-2 py-0.5 text-xs"
                    style={{
                      backgroundColor: alert.confidence === "high" ? "#04771B" :
                                     alert.confidence === "medium" ? "#19647E" : "#E8E3D8",
                    }}
                  >
                    {alert.confidence.toUpperCase()} CONFIDENCE
                  </div>
                </div>
              </motion.button>
            );
          })}
        </div>

        {filteredAlerts.length === 0 && (
          <div className="p-8 text-center">
            <p className="text-sm opacity-60">No alerts match this filter</p>
          </div>
        )}
      </div>

      {/* Bottom Info */}
      <div className="border-t-2 border-black bg-[#E8E3D8] p-4">
        <p className="text-xs text-center">
          Tap any event to highlight its location on the map
        </p>
      </div>
    </div>
  );
}