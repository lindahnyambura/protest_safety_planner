// RouteDestination.tsx
import { useState } from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Slider } from './ui/slider';
import { ArrowLeft, Search, MapPin } from 'lucide-react';
import { motion } from 'motion/react';
import type { RouteData } from '../App';

interface RouteDestinationProps {
  onBack: () => void;
  onComputeRoute: (data: RouteData) => void;
}

export default function RouteDestination({ onBack, onComputeRoute }: RouteDestinationProps) {
  console.log('[RouteDestination] Component mounted');
  const [searchQuery, setSearchQuery] = useState('');
  const [riskPreference, setRiskPreference] = useState([50]);

  // Updated with real Nairobi landmarks
  const presets = [
    { label: 'Jamia Mosque', location: 'Jamia Mosque' },
    { label: 'National Archives', location: 'National Archives' },
    { label: 'Afya Center', location: 'Afya Center' },
    { label: 'GPO', location: 'GPO (General Post Office)' },
  ];

  // Updated recent searches
  const recentSearches = [
    'Railway Station',
    'KICC',
    'Jamia Mosque'
  ];

  const handlePresetClick = (location: string) => {
    setSearchQuery(location);
  };

  const handleComputeRoute = () => {
    if (!searchQuery.trim()) return;
    
    const destination = searchQuery;
    
    // Generate UI-level route data (backend will override most values)
    const riskLevel = riskPreference[0] < 35 ? 'low' : riskPreference[0] > 65 ? 'high' : 'medium';

    onComputeRoute({
      destination,
      safetyScore: 0,
      eta: 0,
      distance: 0,
      riskLevel,
    });
  };

  const canCompute = searchQuery.trim().length > 0;

  return (
    <div className="h-full flex flex-col" style={{ backgroundColor: '#e6e6e6' }}>
      {/* Header */}
      <div className="px-6 py-4 border-b border-neutral-200">
        <motion.div 
          className="flex items-center gap-3 mb-4"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Button
            variant="ghost"
            size="icon"
            onClick={onBack}
            className="rounded-full"
            asChild
          >
            <motion.button whileTap={{ scale: 0.9 }}>
              <ArrowLeft className="w-5 h-5" strokeWidth={2} />
            </motion.button>
          </Button>
          <h2>Find Safe Route</h2>
        </motion.div>

        {/* Search Bar */}
        <motion.div 
          className="relative"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-neutral-400" strokeWidth={2} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search destination..."
            className="w-full pl-12 pr-4 py-3 border-2 border-neutral-300 rounded-xl focus:outline-none focus:border-neutral-900 transition-colors"
          />
        </motion.div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-6 py-6">
        {/* Preset Chips */}
        <motion.div 
          className="mb-8"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <label className="text-sm text-neutral-600 mb-3 block">Quick destinations</label>
          <div className="grid grid-cols-2 gap-2">
            {presets.map((preset, idx) => (
              <motion.button
                key={idx}
                onClick={() => handlePresetClick(preset.location)}
                className="flex items-center justify-center gap-2 px-4 py-3 bg-white text-neutral-700 border-2 border-neutral-300 hover:border-neutral-900 rounded-xl transition-colors"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <MapPin className="w-4 h-4 text-black" strokeWidth={2} />
                <span className="text-sm font-medium">{preset.label}</span>
              </motion.button>
            ))}
          </div>
        </motion.div>

        {/* Risk vs Distance Slider */}
        <motion.div 
          className="mb-8"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="flex items-center justify-between mb-3">
            <label className="text-sm text-neutral-600">Route preference</label>
            <span className="text-sm text-neutral-900 font-medium">
              {riskPreference[0] < 35 ? 'Safest' : riskPreference[0] > 65 ? 'Shortest' : 'Balanced'}
            </span>
          </div>
          
          <Slider
            value={riskPreference}
            onValueChange={setRiskPreference}
            max={100}
            step={1}
            className="mb-2"
          />
          
          <div className="flex justify-between text-xs text-neutral-500">
            <span>Safest route</span>
            <span>Shortest distance</span>
          </div>

          <div className="mt-4 bg-neutral-50 rounded-xl p-4 border-2 border-neutral-200">
            <p className="text-sm text-neutral-700">
              {riskPreference[0] < 35 
                ? 'Prioritizes avoiding risk areas, even if the route is longer.'
                : riskPreference[0] > 65
                ? 'Takes the shortest path with less consideration for risk.'
                : 'Balances safety and distance for an optimal route.'
              }
            </p>
          </div>
        </motion.div>

        {/* Recent Searches */}
        <motion.div /* ... */>
          <label className="text-sm text-neutral-600 mb-3 block">Recent searches</label>
          <div className="space-y-2">
            {recentSearches.map((location, idx) => (
              <motion.button
                key={idx}
                onClick={() => setSearchQuery(location)}
                className="w-full flex items-center gap-3 p-3 bg-white border-2 border-neutral-200 rounded-xl hover:border-neutral-900 transition-colors"
                whileHover={{ scale: 1.01 }}
                whileTap={{ scale: 0.99 }}
              >
                <MapPin className="w-4 h-4 text-neutral-400" strokeWidth={2} />
                <span className="text-sm text-neutral-700">{location}</span>
              </motion.button>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Bottom Action */}
      <div className="px-6 py-4 border-t border-neutral-200">
        <Button
          onClick={handleComputeRoute}
          className={`w-full ${
            canCompute 
              ? 'bg-neutral-900 hover:bg-neutral-800' 
              : 'bg-neutral-300 cursor-not-allowed'
          }`}
          size="lg"
          disabled={!canCompute}
          asChild
        >
          <motion.button
            whileHover={canCompute ? { scale: 1.02 } : {}}
            whileTap={canCompute ? { scale: 0.98 } : {}}
          >
            Compute Safe Route
          </motion.button>
        </Button>
      </div>
    </div>
  );
}
