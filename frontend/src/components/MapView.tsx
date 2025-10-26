import { MapPin, AlertTriangle, Wind, Users } from "lucide-react";

interface HazardMarker {
  id: string;
  type: "gas" | "police" | "crowd" | "safe";
  x: number;
  y: number;
}

interface MapViewProps {
  showUserLocation?: boolean;
  showHazards?: boolean;
  showRoute?: boolean;
  className?: string;
}

export function MapView({ 
  showUserLocation = true, 
  showHazards = true,
  showRoute = false,
  className = "" 
}: MapViewProps) {
  const hazards: HazardMarker[] = [
    { id: "1", type: "gas", x: 45, y: 40 },
    { id: "2", type: "police", x: 65, y: 55 },
    { id: "3", type: "crowd", x: 30, y: 75 },
  ];

  const getHazardColor = (type: string) => {
    switch (type) {
      case "gas": return "#AE1E2A";
      case "police": return "#19647E";
      case "crowd": return "#E8E3D8";
      default: return "#04771B";
    }
  };

  const getHazardIcon = (type: string) => {
    switch (type) {
      case "gas": return <Wind size={16} className="text-black" />;
      case "police": return <AlertTriangle size={16} className="text-black" />;
      case "crowd": return <Users size={16} className="text-black" />;
      default: return null;
    }
  };

  return (
    <div className={`relative w-full h-full bg-[#E8E3D8] border-2 border-black ${className}`}>
      {/* Grid pattern for map aesthetic */}
      <div 
        className="absolute inset-0 opacity-20" 
        style={{
          backgroundImage: `
            linear-gradient(#000 1px, transparent 1px),
            linear-gradient(90deg, #000 1px, transparent 1px)
          `,
          backgroundSize: '20px 20px'
        }}
      />

      {/* Street-like patterns */}
      <div className="absolute top-1/4 left-0 right-0 h-[2px] bg-black opacity-30" />
      <div className="absolute top-2/3 left-0 right-0 h-[2px] bg-black opacity-30" />
      <div className="absolute left-1/3 top-0 bottom-0 w-[2px] bg-black opacity-30" />
      <div className="absolute left-2/3 top-0 bottom-0 w-[2px] bg-black opacity-30" />

      {/* Route polyline */}
      {showRoute && (
        <svg className="absolute inset-0 w-full h-full" style={{ zIndex: 1 }}>
          <path
            d="M 20% 80% L 35% 65% L 50% 50% L 65% 40% L 80% 25%"
            stroke="#6B7F59"
            strokeWidth="4"
            fill="none"
            strokeDasharray="8,4"
          />
        </svg>
      )}

      {/* Hazard markers */}
      {showHazards && hazards.map((hazard) => (
        <div
          key={hazard.id}
          className="absolute -translate-x-1/2 -translate-y-1/2 border-2 border-black p-2"
          style={{
            left: `${hazard.x}%`,
            top: `${hazard.y}%`,
            backgroundColor: getHazardColor(hazard.type),
            zIndex: 2
          }}
        >
          {getHazardIcon(hazard.type)}
        </div>
      ))}

      {/* User location marker */}
      {showUserLocation && (
        <div
          className="absolute -translate-x-1/2 -translate-y-1/2 border-3 border-black p-2 bg-[#FDF8F0]"
          style={{ left: '50%', top: '60%', zIndex: 3 }}
        >
          <MapPin size={20} className="text-black" fill="#000" />
        </div>
      )}
    </div>
  );
}