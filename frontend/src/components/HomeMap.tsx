import MapboxMap from './MapboxMap';
import { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { MapPin, Plus, Navigation, Bell, Settings, Layers, RefreshCw } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface HomeMapProps {
  userLocation: string;
  userCoords: { lat: number; lng: number } | null;
  onReport: () => void;
  onFindRoute: () => void;
  onAlerts: () => void;
  onSettings: () => void;
  onMapClick?: (lat: number, lng: number) => void;
}

export interface ReportMarker {
  id: string;
  type: 'safe' | 'crowd' | 'police' | 'tear_gas' | 'water_cannon';
  lat: number;
  lng: number;
  confidence: number;
  timestamp: number;
  expires_at: number;
  node_id: string;
}

// Static medical stations
const MEDICAL_STATIONS = [
  { id: 'jamia', name: 'Jamia Mosque', lat: -1.283635, lng: 36.820671 },
  { id: 'archives', name: 'National Archives', lat: -1.284948, lng: 36.825943 },
  { id: 'uhuru', name: 'near Uhuru Park', lat: -1.286311, lng: 36.817393 }
];

export default function HomeMap({ 
  userLocation,
  userCoords,
  onReport, 
  onFindRoute,
  onAlerts,
  onSettings,
  onMapClick
}: HomeMapProps) {
  const [map, setMap] = useState<mapboxgl.Map | null>(null);
  const [activeLayers, setActiveLayers] = useState<string[]>(['risk']);
  const [showLayers, setShowLayers] = useState(false);
  const [reports, setReports] = useState<ReportMarker[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const API_BASE_URL = import.meta.env.VITE_API_URL;

  // Fetch reports on mount and periodically
  useEffect(() => {
    fetchReports();
    const interval = setInterval(fetchReports, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchReports = async () => {
    try {
      setRefreshing(true);
      const response = await fetch(`${API_BASE_URL}/reports/active`);
      
      if (response.ok) {
        const data = await response.json();
        console.log('[HomeMap] Fetched reports:', data.reports?.length || 0);
        setReports(data.reports || []);
      }
    } catch (error) {
      console.error('[HomeMap] Failed to fetch reports:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const toggleLayer = (layer: string) => {
    setActiveLayers(prev =>
      prev.includes(layer)
        ? prev.filter(l => l !== layer)
        : [...prev, layer]
    );
  };

  const getLayerReports = (layerType: string): ReportMarker[] => {
    return reports.filter(r => r.type === layerType);
  };

  const layers = [
    { id: 'risk', label: 'Risk Zones', color: 'red', reportCount: 0 },
    { id: 'crowd', label: 'Crowd Density', color: 'amber', reportCount: getLayerReports('crowd').length },
    { id: 'police', label: 'Police Activity', color: 'blue', reportCount: getLayerReports('police').length },
    { id: 'tear_gas', label: 'Tear Gas', color: 'red', reportCount: getLayerReports('tear_gas').length },
    { id: 'water_cannon', label: 'Water Cannon', color: 'red', reportCount: getLayerReports('water_cannon').length },
    { id: 'medical', label: 'Medical Stations', color: 'green', reportCount: MEDICAL_STATIONS.length },
    { id: 'safe', label: 'Safe Zones', color: 'green', reportCount: getLayerReports('safe').length }
  ];

  return (
    <div className="h-full relative">
      {/* Mapbox Map */}
      <MapboxMap
        onMapLoad={setMap}
        userLocation={userCoords ? [userCoords.lng, userCoords.lat] : undefined}
        showRiskLayer={activeLayers.includes('risk')}
        reports={reports}
        activeLayers={activeLayers}
        medicalStations={MEDICAL_STATIONS}
        onMapClick={onMapClick}
      />

      {/* Top Bar */}
      <div className="absolute top-0 left-0 right-0 p-4 z-10">
        <div className="flex items-start justify-between">
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            <Button
              variant="outline"
              size="icon"
              onClick={() => setShowLayers(!showLayers)}
              className="bg-white/95 backdrop-blur border-2 border-neutral-200 shadow-lg"
              asChild
            >
              <motion.button whileTap={{ scale: 0.95 }}>
                <Layers className="w-5 h-5 text-black" strokeWidth={2} />
              </motion.button>
            </Button>
          </motion.div>

          <div className="flex flex-col gap-2 items-end">
            <motion.div 
              className="flex gap-2"
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              <Button
                variant="outline"
                size="icon"
                onClick={fetchReports}
                className={`bg-white/95 backdrop-blur border-2 border-neutral-200 shadow-lg ${
                  refreshing ? 'pointer-events-none' : ''
                }`}
                asChild
              >
                <motion.button 
                  whileTap={{ scale: 0.95 }}
                  animate={refreshing ? { rotate: 360 } : {}}
                  transition={refreshing ? { duration: 1, repeat: Infinity, ease: 'linear' } : {}}
                >
                  <RefreshCw className="w-5 h-5 text-black" strokeWidth={2} />
                </motion.button>
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={onAlerts}
                className="bg-white/95 backdrop-blur border-2 border-neutral-200 shadow-lg relative"
                asChild
              >
                <motion.button whileTap={{ scale: 0.95 }}>
                  <Bell className="w-5 h-5 text-black" strokeWidth={2} />
                  {reports.length > 0 && (
                    <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                      {reports.length > 9 ? '9+' : reports.length}
                    </span>
                  )}
                </motion.button>
              </Button>
              <Button
                variant="outline"
                size="icon"
                onClick={onSettings}
                className="bg-white/95 backdrop-blur border-2 border-neutral-200 shadow-lg"
                asChild
              >
                <motion.button whileTap={{ scale: 0.95 }}>
                  <Settings className="w-5 h-5 text-black" strokeWidth={2} />
                </motion.button>
              </Button>
            </motion.div>

            {/* Location Info Card */}
            <motion.div 
              className="bg-white/95 backdrop-blur rounded-xl px-3 py-2 border-2 border-neutral-200 shadow-lg"
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4 text-black" strokeWidth={2} />
                <span className="text-sm text-neutral-900">
                  {userLocation || 'Tap map to set location'}
                </span>
              </div>
              {userCoords && (
                <div className="text-xs text-neutral-500 mt-1">
                  {userCoords.lat.toFixed(4)}°, {userCoords.lng.toFixed(4)}°
                </div>
              )}
            </motion.div>
          </div>
        </div>

        {/* Layer toggles */}
        <AnimatePresence>
          {showLayers && (
            <motion.div 
              className="mt-3 bg-white/95 backdrop-blur rounded-xl p-4 border-2 border-neutral-200 shadow-lg"
              initial={{ opacity: 0, y: -10, height: 0 }}
              animate={{ opacity: 1, y: 0, height: 'auto' }}
              exit={{ opacity: 0, y: -10, height: 0 }}
              transition={{ duration: 0.2 }}
            >
              <p className="text-sm text-neutral-600 mb-3">Map Layers</p>
              <div className="space-y-2">
                {layers.map((layer) => (
                  <motion.button
                    key={layer.id}
                    onClick={() => toggleLayer(layer.id)}
                    className={`w-full text-left px-3 py-2 rounded-lg transition-colors flex items-center justify-between ${
                      activeLayers.includes(layer.id)
                        ? 'bg-neutral-900 text-white'
                        : 'bg-neutral-50 text-neutral-700 hover:bg-neutral-100'
                    }`}
                    whileTap={{ scale: 0.98 }}
                  >
                    <span className="text-sm">{layer.label}</span>
                    {layer.reportCount > 0 && (
                      <span className={`text-xs px-2 py-0.5 rounded-full ${
                        activeLayers.includes(layer.id)
                          ? 'bg-white text-neutral-900'
                          : 'bg-neutral-200 text-neutral-700'
                      }`}>
                        {layer.reportCount}
                      </span>
                    )}
                  </motion.button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Bottom Actions */}
      <div className="absolute bottom-0 left-0 right-0 p-4 z-10 pointer-events-none">
        <motion.div 
          className="flex gap-3 pointer-events-auto"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <Button
            onClick={onReport}
            size="lg"
            className="flex-1 bg-neutral-900 hover:bg-neutral-800 shadow-lg"
            disabled={!userCoords}
            asChild
          >
            <motion.button whileTap={{ scale: 0.98 }}>
              <Plus className="w-5 h-5 mr-2" strokeWidth={2} />
              Report
            </motion.button>
          </Button>
          <Button
            onClick={onFindRoute}
            size="lg"
            variant="outline"
            className="flex-1 bg-white/95 backdrop-blur border-2 border-neutral-200 shadow-lg"
            disabled={!userCoords}
            asChild
          >
            <motion.button whileTap={{ scale: 0.98 }}>
              <Navigation className="w-5 h-5 mr-2" strokeWidth={2} />
              Find Safe Route
            </motion.button>
          </Button>
        </motion.div>

        {!userCoords && (
          <motion.p 
            className="text-center text-sm text-neutral-600 mt-2 pointer-events-none"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            Tap anywhere on the map to set your location
          </motion.p>
        )}
      </div>
    </div>
  );
}