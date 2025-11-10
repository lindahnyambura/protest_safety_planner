import { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { ArrowLeft, MapPin, Search, TrendingDown, TrendingUp, Minus } from 'lucide-react';
import { motion } from 'motion/react';
import { toast } from 'sonner';
import type { RouteData } from '../App';

interface RouteDestinationProps {
  onBack: () => void;
  onComputeRoute: (routeData: RouteData) => void;
}

const NAIROBI_BOUNDS = {
  north: -1.280,
  south: -1.295,
  east: 36.835,
  west: 36.810,
};

const isWithinBounds = (lat: number, lng: number): boolean => {
  return (
    lat >= NAIROBI_BOUNDS.south &&
    lat <= NAIROBI_BOUNDS.north &&
    lng >= NAIROBI_BOUNDS.west &&
    lng <= NAIROBI_BOUNDS.east
  );
};

export default function RouteDestination({ onBack, onComputeRoute }: RouteDestinationProps) {
  const [destination, setDestination] = useState('');
  const [landmarks, setLandmarks] = useState<any[]>([]);
  const [recentSearches] = useState<string[]>([]);
  const [riskPreference, setRiskPreference] = useState<'low' | 'medium' | 'high'>('medium');
  const [searching, setSearching] = useState(false);
  const [loading, setLoading] = useState(false);

  const API_BASE_URL = import.meta.env.VITE_API_URL;

  // Fetch popular landmarks
  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE_URL}/landmarks?limit=8`)
      .then(res => res.json())
      .then(data => {
        console.log('[RouteDestination] Loaded landmarks:', data.landmarks);
        setLandmarks(data.landmarks || []);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load landmarks:', err);
        toast.error('Failed to load destinations');
        setLoading(false);
      });
  }, []);

  const geocodeDestination = async (query: string): Promise<{ name: string; lat: number; lng: number } | null> => {
    try {
      // Use Nominatim to geocode
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query + ', Nairobi CBD, Kenya')}&format=json&limit=1`,
        {
          headers: { 'User-Agent': 'ProtestSafetyPlanner/1.0' }
        }
      );

      if (!response.ok) throw new Error('Geocoding failed');

      const data = await response.json();
      
      if (data.length === 0) {
        toast.error('Location not found', {
          description: 'Try selecting from popular destinations'
        });
        return null;
      }

      const result = data[0];
      const lat = parseFloat(result.lat);
      const lng = parseFloat(result.lon);

      console.log('[RouteDestination] Geocoded:', query, { lat, lng });

      // Check bounds
      if (!isWithinBounds(lat, lng)) {
        toast.error('Destination outside Nairobi CBD', {
          description: `Must be within bounds: ${NAIROBI_BOUNDS.south}°S to ${NAIROBI_BOUNDS.north}°S`
        });
        return null;
      }

      const locationName = result.display_name.split(',')[0] || query;
      return { name: locationName, lat, lng };

    } catch (error) {
      console.error('Geocoding error:', error);
      toast.error('Failed to find destination');
      return null;
    }
  };

  const handleComputeRoute = async (destinationName: string, destLat?: number, destLng?: number) => {
    setSearching(true);

    try {
      let finalDestination = { name: destinationName, lat: destLat || 0, lng: destLng || 0 };

      // If no coordinates provided, geocode the destination
      if (!destLat || !destLng) {
        const geocoded = await geocodeDestination(destinationName);
        if (!geocoded) {
          setSearching(false);
          return;
        }
        finalDestination = geocoded;
      }

      console.log('[RouteDestination] Computing route to:', finalDestination);

      // Create route data with proper risk level mapping
      const routeData: RouteData = {
        destination: finalDestination.name,
        safetyScore: 0, // Will be filled by backend
        eta: 0,
        distance: 0,
        riskLevel: riskPreference, // This determines lambda_risk in backend
      };

      onComputeRoute(routeData);

    } catch (error) {
      console.error('Route computation error:', error);
      toast.error('Failed to compute route');
    } finally {
      setSearching(false);
    }
  };

  const riskOptions = [
    {
      id: 'low' as const,
      label: 'Safest',
      description: 'Avoid all risks',
      icon: TrendingDown,
      color: 'green',
    },
    {
      id: 'medium' as const,
      label: 'Balanced',
      description: 'Balance safety & speed',
      icon: Minus,
      color: 'amber',
    },
    {
      id: 'high' as const,
      label: 'Shortest',
      description: 'Fastest route',
      icon: TrendingUp,
      color: 'blue',
    },
  ];

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <motion.div
        className="px-6 py-4 border-b border-neutral-200"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-3 mb-4">
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
          <div className="flex-1">
            <h3>Choose Destination</h3>
          </div>
        </div>

        {/* Search Input */}
        <div className="relative">
          <input
            type="text"
            value={destination}
            onChange={(e) => setDestination(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && destination.trim()) {
                handleComputeRoute(destination.trim());
              }
            }}
            placeholder="Search for a place..."
            className="w-full px-4 py-3 pl-10 border-2 border-neutral-300 rounded-xl focus:outline-none focus:border-neutral-900 transition-colors"
            disabled={searching}
          />
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-400" />
        </div>

        {destination.trim() && (
          <Button
            onClick={() => handleComputeRoute(destination.trim())}
            className="w-full mt-3 bg-neutral-900 hover:bg-neutral-800"
            disabled={searching}
          >
            {searching ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                Searching...
              </>
            ) : (
              <>
                <Search className="w-4 h-4 mr-2" />
                Find Route
              </>
            )}
          </Button>
        )}
      </motion.div>

      <div className="flex-1 overflow-y-auto px-6 py-4">
        {/* Route Preference */}
        <div className="mb-6">
          <label className="text-sm text-neutral-600 mb-3 block">Route Preference</label>
          <div className="space-y-2">
            {riskOptions.map((option) => {
              const Icon = option.icon;
              return (
                <motion.button
                  key={option.id}
                  onClick={() => setRiskPreference(option.id)}
                  className={`w-full p-4 rounded-xl border-2 transition-all text-left ${
                    riskPreference === option.id
                      ? option.color === 'green'
                        ? 'border-green-500 bg-green-50'
                        : option.color === 'amber'
                        ? 'border-amber-500 bg-amber-50'
                        : 'border-blue-500 bg-blue-50'
                      : 'border-neutral-200 bg-white hover:border-neutral-300'
                  }`}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="flex items-center gap-3">
                    <Icon
                      className={`w-5 h-5 ${
                        riskPreference === option.id ? 'text-neutral-900' : 'text-neutral-500'
                      }`}
                      strokeWidth={2.5}
                    />
                    <div className="flex-1">
                      <div className="font-medium text-neutral-900">{option.label}</div>
                      <div className="text-sm text-neutral-600">{option.description}</div>
                    </div>
                  </div>
                </motion.button>
              );
            })}
          </div>
        </div>

        {/* Popular Destinations */}
        <div className="mb-6">
          <label className="text-sm text-neutral-600 mb-3 block">Popular Destinations</label>
          {loading ? (
            <div className="text-center py-8 text-neutral-500">Loading destinations...</div>
          ) : (
            <div className="space-y-2">
              {landmarks.map((landmark, idx) => (
                <button
                  key={idx}
                  onClick={() =>
                    handleComputeRoute(
                      landmark.name,
                      landmark.coordinates.lat,
                      landmark.coordinates.lng
                    )
                  }
                  disabled={searching}
                  className="w-full px-4 py-3 bg-white border-2 border-neutral-200 rounded-xl hover:border-neutral-900 transition-colors text-left disabled:opacity-50"
                >
                  <div className="flex items-center gap-3">
                    <MapPin className="w-4 h-4 text-neutral-400 flex-shrink-0" />
                    <div className="flex-1">
                      <div className="text-sm font-medium text-neutral-900">{landmark.name}</div>
                      <div className="text-xs text-neutral-500">
                        {landmark.coordinates.lat.toFixed(4)}°, {landmark.coordinates.lng.toFixed(4)}°
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Recent Searches */}
        {recentSearches.length > 0 && (
          <div>
            <label className="text-sm text-neutral-600 mb-3 block">Recent Searches</label>
            <div className="space-y-2">
              {recentSearches.map((search, idx) => (
                <button
                  key={idx}
                  onClick={() => handleComputeRoute(search)}
                  disabled={searching}
                  className="w-full px-4 py-3 bg-neutral-50 border-2 border-neutral-200 rounded-xl hover:border-neutral-900 transition-colors text-left disabled:opacity-50"
                >
                  <div className="flex items-center gap-3">
                    <MapPin className="w-4 h-4 text-neutral-400" />
                    <span className="text-sm text-neutral-900">{search}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}