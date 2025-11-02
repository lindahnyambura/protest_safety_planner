// frontend/src/components/HomeMap.tsx
import { useState, useEffect, useRef } from "react";
import { motion } from "motion/react";
import { AlertCircle, MapPin, Route, Bell, Settings as SettingsIcon, ChevronUp, ChevronDown, ArrowLeft } from "lucide-react";
import { MapView } from "./MapView";
import type { Screen } from "../App";
import { apiClient } from "../lib/api";

interface Event {
  id: string;
  type: string;
  location: string;
  time: string;
  severity: "low" | "medium" | "high";
}

interface HomeMapProps {
  onNavigate: (screen: string, data?: any) => void;
  onBack?: () => void;
}

export function HomeMap({ onNavigate, onBack }: HomeMapProps) {
  const [eventsExpanded, setEventsExpanded] = useState(false);
  const [riskLevel, setRiskLevel] = useState<"low" | "moderate" | "high">("low");
  const [userLocation, setUserLocation] = useState<{ lat: number; lng: number } | null>(null);
  const [locationName, setLocationName] = useState<string>("Locating...");
  const [alerts, setAlerts] = useState<Event[]>([]);

  // Load user location and risk data on component mount
  useEffect(() => {
    loadUserLocation();
    loadAlerts();
    determineRiskLevel();
  }, []);

  const loadUserLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setUserLocation({ lat: latitude, lng: longitude });
          reverseGeocode(latitude, longitude);
        },
        (error) => {
          console.warn("Geolocation failed:", error);
          // Fallback to Nairobi CBD center
          setUserLocation({ lat: -1.2833, lng: 36.8172 });
          setLocationName("Nairobi CBD");
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
          maximumAge: 60000
        }
      );
    } else {
      // Fallback if geolocation not supported
      setUserLocation({ lat: -1.2833, lng: 36.8172 });
      setLocationName("Nairobi CBD");
    }
  };

  const reverseGeocode = async (lat: number, lng: number) => {
    try {
      // Simple reverse geocoding - in production you'd use MapBox Geocoding API
      const landmarks = [
        { name: "Uhuru Park", lat: -1.2921, lng: 36.8219, threshold: 0.005 },
        { name: "Central Police Station", lat: -1.2836, lng: 36.8221, threshold: 0.003 },
        { name: "Kenyatta International Conference Centre", lat: -1.2889, lng: 36.8233, threshold: 0.004 },
        { name: "Nairobi Central", lat: -1.2833, lng: 36.8172, threshold: 0.01 }
      ];

      let closestLandmark = "Nairobi CBD";
      let minDistance = Infinity;

      landmarks.forEach(landmark => {
        const distance = Math.sqrt(
          Math.pow(lat - landmark.lat, 2) + Math.pow(lng - landmark.lng, 2)
        );
        if (distance < minDistance && distance < landmark.threshold) {
          minDistance = distance;
          closestLandmark = landmark.name;
        }
      });

      setLocationName(closestLandmark);
    } catch (error) {
      console.error("Reverse geocoding failed:", error);
      setLocationName("Nairobi CBD");
    }
  };

  const loadAlerts = async () => {
    try {
      const alertsData = await apiClient.getAlerts();
      setAlerts(alertsData);
    } catch (error) {
      console.error("Failed to load alerts:", error);
      // Fallback to mock data
      setAlerts([
        { id: "1", type: "Tear gas reported", location: "Tom Mboya St", time: "2 min ago", severity: "high" },
        { id: "2", type: "Heavy crowd", location: "University Way", time: "8 min ago", severity: "medium" },
        { id: "3", type: "Police presence", location: "Kenyatta Ave", time: "15 min ago", severity: "medium" },
      ]);
    }
  };

  const determineRiskLevel = async () => {
    try {
      const riskData = await apiClient.getRiskMap();
      
      // Simple risk calculation based on nearby risk points
      if (userLocation) {
        let nearbyRisk = 0;
        let riskCount = 0;
        
        riskData.features.forEach((feature: any) => {
          const riskLat = feature.geometry.coordinates[1];
          const riskLng = feature.geometry.coordinates[0];
          
          // Calculate distance (simplified)
          const distance = Math.sqrt(
            Math.pow(userLocation.lat - riskLat, 2) + 
            Math.pow(userLocation.lng - riskLng, 2)
          );
          
          if (distance < 0.002) { // ~200 meter radius
            nearbyRisk += feature.properties.risk;
            riskCount++;
          }
        });
        
        const avgRisk = riskCount > 0 ? nearbyRisk / riskCount : 0;
        
        if (avgRisk > 0.3) setRiskLevel("high");
        else if (avgRisk > 0.1) setRiskLevel("moderate");
        else setRiskLevel("low");
      }
    } catch (error) {
      console.error("Failed to determine risk level:", error);
      // Default to low risk if determination fails
      setRiskLevel("low");
    }
  };

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
        <div className="flex items-center gap-2">
          {onBack && (
            <button 
              onClick={onBack}
              className="border-2 border-black p-2 hover:bg-[#E8E3D8] transition-colors"
            >
              <ArrowLeft size={18} />
            </button>
          )}
          <button 
            onClick={() => onNavigate("settings")}
            className="border-2 border-black p-2 hover:bg-[#E8E3D8] transition-colors"
          >
            <SettingsIcon size={18} />
          </button>
        </div>
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
        <MapView 
          showUserLocation 
          showHazards 
          userLocation={userLocation}
          onLocationUpdate={(location, name) => {
            setUserLocation(location);
            setLocationName(name);
            determineRiskLevel();
          }}
        />

        {/* Risk Status Card */}
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

        {/* Current Location Label */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute top-4 right-4 border-2 border-black bg-[#FDF8F0] px-3 py-1.5 text-xs max-w-[40%]"
        >
          <div className="flex items-center gap-1.5">
            <MapPin size={12} />
            <span className="truncate">{locationName}</span>
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
          {alerts.map((event, index) => (
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