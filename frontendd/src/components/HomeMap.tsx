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
  const [activeLayers, setActiveLayers] = useState<string[]>(['risk', 'crowd']);
  const [showLayers, setShowLayers] = useState(false);

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
    <div className="h-full relative bg-neutral-100">
      {/* Mock Map */}
      <div className="absolute inset-0 bg-gradient-to-br from-neutral-100 to-neutral-200">
        {/* Grid pattern for map effect */}
        <div className="absolute inset-0 opacity-10">
          {[...Array(20)].map((_, i) => (
            <div key={`h-${i}`} className="absolute w-full h-px bg-neutral-400" style={{ top: `${i * 5}%` }} />
          ))}
          {[...Array(20)].map((_, i) => (
            <div key={`v-${i}`} className="absolute h-full w-px bg-neutral-400" style={{ left: `${i * 5}%` }} />
          ))}
        </div>

        {/* Incident markers with gradient effect */}
        {incidents.map((incident, idx) => (
          <motion.div
            key={idx}
            className="absolute transform -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${incident.x}%`, top: `${incident.y}%` }}
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ delay: idx * 0.1, type: 'spring', stiffness: 200 }}
          >
            {/* Gradient background */}
            <motion.div
              className={`absolute inset-0 rounded-full blur-2xl ${
                incident.type === 'peaceful' || incident.type === 'safe'
                  ? 'bg-green-400'
                  : incident.type === 'police'
                  ? 'bg-blue-400'
                  : 'bg-amber-400'
              }`}
              style={{
                width: incident.size === 'large' ? '120px' : incident.size === 'medium' ? '80px' : '50px',
                height: incident.size === 'large' ? '120px' : incident.size === 'medium' ? '80px' : '50px',
                opacity: incident.intensity * 0.3,
              }}
              animate={{
                scale: [1, 1.2, 1],
                opacity: [incident.intensity * 0.3, incident.intensity * 0.2, incident.intensity * 0.3],
              }}
              transition={{
                duration: 3,
                repeat: Infinity,
                ease: 'easeInOut',
              }}
            />
            
            {/* Icon marker */}
            <div
              className={`relative z-10 rounded-full border-3 bg-white shadow-lg flex items-center justify-center ${
                incident.type === 'peaceful' || incident.type === 'safe'
                  ? 'border-green-500'
                  : incident.type === 'police'
                  ? 'border-blue-500'
                  : 'border-amber-500'
              } ${
                incident.size === 'large'
                  ? 'w-12 h-12'
                  : incident.size === 'medium'
                  ? 'w-8 h-8'
                  : 'w-6 h-6'
              }`}
            >
              <MapPin 
                className={`${
                  incident.size === 'large' ? 'w-6 h-6' : incident.size === 'medium' ? 'w-4 h-4' : 'w-3 h-3'
                } text-neutral-900`}
                strokeWidth={2}
              />
            </div>
          </motion.div>
        ))}

        {/* User location marker */}
        <div
          className="absolute transform -translate-x-1/2 -translate-y-1/2 z-20"
          style={{ left: '50%', top: '50%' }}
        >
          <div className="relative">
            <div className="w-4 h-4 bg-neutral-900 rounded-full border-2 border-white shadow-lg" />
            <motion.div 
              className="absolute inset-0 w-4 h-4 bg-neutral-900 rounded-full"
              animate={{
                scale: [1, 2.5, 1],
                opacity: [0.6, 0, 0.6],
              }}
              transition={{
                duration: 2.5,
                repeat: Infinity,
                ease: 'easeOut',
              }}
            />
          </div>
        </div>
      </div>

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
