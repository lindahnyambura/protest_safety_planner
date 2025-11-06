import MapboxMap from './MapboxMap';
import { useState } from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { MapPin, Plus, Navigation, Bell, Settings, Layers } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface HomeMapProps {
  userLocation: string;
  onReport: () => void;
  onFindRoute: () => void;
  onAlerts: () => void;
  onSettings: () => void;
}

export default function HomeMap({ 
  userLocation, 
  onReport, 
  onFindRoute,
  onAlerts,
  onSettings 
}: HomeMapProps) {
  // Map related states
  const [map, setMap] = useState<mapboxgl.Map | null>(null);
  const [userCoords, setUserCoords] = useState<[number, number]>([36.8225, -1.2875]);

  // UI related states
  const [activeLayers, setActiveLayers] = useState<string[]>(['risk', 'crowd']);
  const [showLayers, setShowLayers] = useState(true);

  const toggleLayer = (layer: string) => {
    setActiveLayers(prev =>
      prev.includes(layer)
        ? prev.filter(l => l !== layer)
        : [...prev, layer]
    );
  };

  const incidents = [
    { x: 35, y: 40, type: 'peaceful', size: 'large', intensity: 0.8 },
    { x: 60, y: 30, type: 'police', size: 'small', intensity: 0.6 },
    { x: 45, y: 70, type: 'crowd', size: 'medium', intensity: 0.7 },
    { x: 70, y: 60, type: 'safe', size: 'small', intensity: 0.5 },
  ];

  return (
    <div className="h-full relative">
      {/* Mapbox Map */}
      <MapboxMap
        onMapLoad={setMap}
        userLocation={userCoords}
        showRiskLayer={activeLayers.includes('risk')}
      />

      {/* Top Bar */}
      <div className="absolute top-0 left-0 right-0 p-4">
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
                onClick={onAlerts}
                className="bg-white/95 backdrop-blur border-2 border-neutral-200 shadow-lg"
                asChild
              >
                <motion.button whileTap={{ scale: 0.95 }}>
                  <Bell className="w-5 h-5 text-black" strokeWidth={2} />
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

            {/* Location Info Card - smaller and repositioned */}
            <motion.div 
              className="bg-white/95 backdrop-blur rounded-xl px-3 py-2 border-2 border-neutral-200 shadow-lg"
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4 text-black" strokeWidth={2} />
                <span className="text-sm text-neutral-900">Kenyatta Ave</span>
              </div>
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
                {[
                  { id: 'risk', label: 'Risk Zones' },
                  { id: 'crowd', label: 'Crowd Density' },
                  { id: 'police', label: 'Police Activity' },
                  { id: 'medical', label: 'Medical Stations' },
                  { id: 'safe', label: 'Safe Zones' }
                ].map((layer) => (
                  <motion.button
                    key={layer.id}
                    onClick={() => toggleLayer(layer.id)}
                    className={`w-full text-left px-3 py-2 rounded-lg transition-colors ${
                      activeLayers.includes(layer.id)
                        ? 'bg-neutral-900 text-white'
                        : 'bg-neutral-50 text-neutral-700 hover:bg-neutral-100'
                    }`}
                    whileTap={{ scale: 0.98 }}
                  >
                    <span className="text-sm">{layer.label}</span>
                  </motion.button>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Bottom Actions */}
      <div className="absolute bottom-0 left-0 right-0 p-4">
        <motion.div 
          className="flex gap-3"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.4 }}
        >
          <Button
            onClick={onReport}
            size="lg"
            className="flex-1 bg-neutral-900 hover:bg-neutral-800 shadow-lg"
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
            asChild
          >
            <motion.button whileTap={{ scale: 0.98 }}>
              <Navigation className="w-5 h-5 mr-2" strokeWidth={2} />
              Find Safe Route
            </motion.button>
          </Button>
        </motion.div>
      </div>
    </div>
  );
}
