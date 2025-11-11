// RouteDestination.tsx
import { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Slider } from './ui/slider';
import { ArrowLeft, Search, MapPin, Loader2 } from 'lucide-react';
import { motion } from 'motion/react';
import type { RouteData } from '../App';

interface RouteDestinationProps {
  onBack: () => void;
  onComputeRoute: (data: RouteData) => void;
}

interface SearchResult {
  name: string;
  type: 'landmark' | 'street' | 'place';
  node_id: string | null;
  coordinates: { lat: number; lng: number } | null;
}

export default function RouteDestination({ onBack, onComputeRoute }: RouteDestinationProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [riskPreference, setRiskPreference] = useState([50]); // 0=safest, 100=shortest
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showResults, setShowResults] = useState(false);

  const API_BASE_URL = import.meta.env.VITE_API_URL;

  // Quick destination presets
  const presets = [
    { label: 'Jamia Mosque', location: 'Jamia Mosque' },
    { label: 'National Archives', location: 'National Archives' },
    { label: 'Afya Center', location: 'Afya Center' },
    { label: 'GPO', location: 'GPO' },
  ];

  const recentSearches = ['Railway Station', 'KICC', 'Jamia Mosque'];

  // Debounced search for destinations
  useEffect(() => {
    if (searchQuery.length < 2) {
      setSearchResults([]);
      setShowResults(false);
      return;
    }

    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const response = await fetch(
          `${API_BASE_URL}/search-destinations?q=${encodeURIComponent(searchQuery)}`
        );
        
        if (response.ok) {
          const data = await response.json();
          setSearchResults(data.results || []);
          setShowResults(data.results.length > 0);
        }
      } catch (error) {
        console.error('Search error:', error);
      } finally {
        setIsSearching(false);
      }
    }, 300); // 300ms debounce

    return () => clearTimeout(timer);
  }, [searchQuery, API_BASE_URL]);

  const handlePresetClick = (location: string) => {
    setSearchQuery(location);
    setShowResults(false);
  };

  const handleResultClick = (result: SearchResult) => {
    setSearchQuery(result.name);
    setShowResults(false);
  };

  const handleComputeRoute = () => {
    if (!searchQuery.trim()) return;
    
    // Map slider position to risk level
    // 0-35: Safest (left side)
    // 35-65: Balanced (middle)
    // 65-100: Shortest (right side)
    const riskLevel = 
      riskPreference[0] < 35 
        ? 'low'    // Safest
        : riskPreference[0] > 65 
        ? 'high'   // Shortest
        : 'medium'; // Balanced

    onComputeRoute({
      destination: searchQuery,
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

        {/* Search Bar with Results Dropdown */}
        <motion.div 
          className="relative"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-neutral-400 z-10" strokeWidth={2} />
          
          {isSearching && (
            <Loader2 className="absolute right-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-neutral-400 animate-spin z-10" />
          )}
          
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onFocus={() => searchResults.length > 0 && setShowResults(true)}
            placeholder="Search destination in Nairobi CBD..."
            className="w-full pl-12 pr-4 py-3 border-2 border-neutral-300 rounded-xl focus:outline-none focus:border-neutral-900 transition-colors"
          />

          {/* Search Results Dropdown */}
          {showResults && searchResults.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="absolute top-full left-0 right-0 mt-2 bg-white border-2 border-neutral-300 rounded-xl shadow-lg max-h-64 overflow-y-auto z-20"
            >
              {searchResults.map((result, idx) => (
                <button
                  key={idx}
                  onClick={() => handleResultClick(result)}
                  className="w-full flex items-center gap-3 px-4 py-3 hover:bg-neutral-50 transition-colors border-b border-neutral-100 last:border-0"
                >
                  <MapPin className="w-4 h-4 text-neutral-400 flex-shrink-0" strokeWidth={2} />
                  <div className="flex-1 text-left">
                    <p className="text-sm text-neutral-900">{result.name}</p>
                    <p className="text-xs text-neutral-500 capitalize">{result.type}</p>
                  </div>
                </button>
              ))}
            </motion.div>
          )}
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
                ? 'Prioritizes avoiding risk areas, even if the route is longer. Uses optimal Dijkstra algorithm.'
                : riskPreference[0] > 65
                ? 'Takes the shortest path with less consideration for risk. Uses fast A* algorithm.'
                : 'Balances safety and distance for an optimal route. Uses A* algorithm.'
              }
            </p>
          </div>
        </motion.div>

        {/* Recent Searches */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
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