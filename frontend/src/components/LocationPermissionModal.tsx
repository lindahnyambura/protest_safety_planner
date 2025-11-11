import { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { MapPin, Info, X, Search } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { toast } from 'sonner';

interface LocationPermissionModalProps {
  onLocationGranted: (location: string, coords: { lat: number; lng: number }) => void;
  onClose: () => void;
}

// Nairobi CBD bounds
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

export default function LocationPermissionModal({ onLocationGranted, onClose }: LocationPermissionModalProps) {
  const [showWhy, setShowWhy] = useState(false);
  const [manualLocation, setManualLocation] = useState('');
  const [useManual, setUseManual] = useState(false);
  const [loading, setLoading] = useState(false);
  const [geocoding, setGeocoding] = useState(false);
  const [landmarks, setLandmarks] = useState<any[]>([]);
  const API_BASE_URL = import.meta.env.VITE_API_URL;

  // Fetch landmarks for manual selection (limit to 3 most popular)
  useEffect(() => {
    if (useManual) {
      setLoading(true);
      fetch(`${API_BASE_URL}/landmarks?limit=3`)
        .then(res => res.json())
        .then(data => {
          console.log('[LocationModal] Loaded landmarks:', data.landmarks);
          setLandmarks(data.landmarks || []);
          setLoading(false);
        })
        .catch(err => {
          console.error('Failed to load landmarks:', err);
          toast.error('Failed to load landmarks');
          setLoading(false);
        });
    }
  }, [useManual]);

  const handleGrantPermission = () => {
    setLoading(true);

    if (!navigator.geolocation) {
      toast.error('Geolocation not supported by your browser');
      setLoading(false);
      setUseManual(true);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const { latitude, longitude } = position.coords;
        
        console.log('[LocationModal] GPS location:', { lat: latitude, lng: longitude });
        
        // Check if within Nairobi CBD bounds
        if (!isWithinBounds(latitude, longitude)) {
          toast.error('Location outside Nairobi CBD area', {
            description: 'Please use manual location selection'
          });
          setLoading(false);
          setUseManual(true);
          return;
        }

        // Reverse geocode to get location name
        fetch(`https://nominatim.openstreetmap.org/reverse?lat=${latitude}&lon=${longitude}&format=json`, {
          headers: { 'User-Agent': 'ProtestSafetyPlanner/1.0' }
        })
          .then(res => res.json())
          .then(data => {
            const locationName = data.address?.road || data.address?.neighbourhood || 'Nairobi CBD';
            console.log('[LocationModal] Location set:', locationName, { lat: latitude, lng: longitude });
            onLocationGranted(locationName, { lat: latitude, lng: longitude });
            setLoading(false);
          })
          .catch(() => {
            console.log('[LocationModal] Location set (no name):', { lat: latitude, lng: longitude });
            onLocationGranted('Nairobi CBD', { lat: latitude, lng: longitude });
            setLoading(false);
          });
      },
      (error) => {
        console.error('Geolocation error:', error);
        toast.error('Could not get your location', {
          description: 'Please use manual location selection'
        });
        setLoading(false);
        setUseManual(true);
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 0
      }
    );
  };

  const handleManualLocation = async (landmark?: any) => {
    if (landmark) {
      // Use exact coordinates from the landmark
      console.log('[LocationModal] Landmark selected:', landmark.name, landmark.coordinates);
      
      // Verify coordinates are within bounds
      if (!isWithinBounds(landmark.coordinates.lat, landmark.coordinates.lng)) {
        toast.error('Location outside Nairobi CBD bounds');
        return;
      }
      
      onLocationGranted(landmark.name, {
        lat: landmark.coordinates.lat,
        lng: landmark.coordinates.lng
      });
    } else if (manualLocation.trim()) {
      // Geocode the typed location
      await geocodeLocation(manualLocation.trim());
    }
  };

  const geocodeLocation = async (query: string) => {
    setGeocoding(true);
    
    try {
      // Use Nominatim to geocode the location
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query + ', Nairobi CBD, Kenya')}&format=json&limit=1`,
        {
          headers: { 'User-Agent': 'ProtestSafetyPlanner/1.0' }
        }
      );

      if (!response.ok) {
        throw new Error('Geocoding failed');
      }

      const data = await response.json();
      
      if (data.length === 0) {
        toast.error('Location not found', {
          description: 'Try selecting from the landmarks list'
        });
        setGeocoding(false);
        return;
      }

      const result = data[0];
      const lat = parseFloat(result.lat);
      const lng = parseFloat(result.lon);

      console.log('[LocationModal] Geocoded location:', query, { lat, lng });

      // Check if within bounds
      if (!isWithinBounds(lat, lng)) {
        toast.error('Location outside Nairobi CBD', {
          description: `Bounds: ${NAIROBI_BOUNDS.south}°S to ${NAIROBI_BOUNDS.north}°S`
        });
        setGeocoding(false);
        return;
      }

      const locationName = result.display_name.split(',')[0] || query;
      onLocationGranted(locationName, { lat, lng });
      
    } catch (error) {
      console.error('Geocoding error:', error);
      toast.error('Failed to find location', {
        description: 'Please try selecting from landmarks'
      });
    } finally {
      setGeocoding(false);
    }
  };

  return (
    <motion.div 
      className="fixed inset-0 z-50 flex items-center justify-center px-6"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      {/* Backdrop */}
      <motion.div 
        className="absolute inset-0 bg-black/40 backdrop-blur-md"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      />

      {/* Modal Content */}
      <motion.div 
        className="relative w-full max-w-md"
        initial={{ scale: 0.9, y: 20, opacity: 0 }}
        animate={{ scale: 1, y: 0, opacity: 1 }}
        exit={{ scale: 0.9, y: 20, opacity: 0 }}
        transition={{ type: 'spring', stiffness: 300, damping: 25 }}
      >
        <div className="bg-white rounded-2xl p-8 shadow-2xl border-2 border-neutral-200 max-h-[90vh] overflow-y-auto">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 p-2 hover:bg-neutral-100 rounded-full transition-colors"
          >
            <X className="w-5 h-5 text-neutral-600" strokeWidth={2} />
          </button>

          <motion.div 
            className="w-16 h-16 mx-auto mb-6 bg-neutral-900 rounded-2xl flex items-center justify-center"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
          >
            <MapPin className="w-8 h-8 text-white" strokeWidth={2} />
          </motion.div>
          
          <h2 className="text-center mb-3">Location Access</h2>
          <p className="text-center text-neutral-600 mb-6">
            SafeNav needs your location to provide real-time safety updates and route guidance.
          </p>

          {!useManual ? (
            // Auto Location Flow
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <Button 
                onClick={handleGrantPermission}
                className="w-full bg-neutral-900 hover:bg-neutral-800 mb-3"
                size="lg"
                disabled={loading}
              >
                {loading ? 'Getting location...' : 'Allow Location Access'}
              </Button>

              <motion.button
                onClick={() => setShowWhy(!showWhy)}
                className="w-full text-sm text-neutral-600 hover:text-neutral-900 flex items-center justify-center gap-2 mb-3 py-2"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Info className="w-4 h-4" strokeWidth={2} />
                Why we need your location
              </motion.button>

              <AnimatePresence>
                {showWhy && (
                  <motion.div 
                    className="bg-neutral-50 rounded-xl p-4 mb-3 text-sm text-neutral-700 border-2 border-neutral-200"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <p className="mb-2">We use your location to:</p>
                    <ul className="space-y-1 ml-4">
                      <li>• Show nearby safety incidents</li>
                      <li>• Calculate safe routes</li>
                      <li>• Provide real-time turn-by-turn guidance</li>
                    </ul>
                    <p className="mt-3 text-neutral-600">
                      Your location is never stored permanently and expires after 10 minutes.
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>

              <motion.button
                onClick={() => setUseManual(true)}
                className="w-full text-sm text-neutral-600 hover:text-neutral-900 py-2"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Use manual location
              </motion.button>
            </motion.div>
          ) : (
            // Manual Location Flow
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
            >
              <p className="text-sm text-neutral-600 mb-3">Select or enter your location:</p>

              {/* Search input */}
              <div className="relative mb-4">
                <input
                  type="text"
                  value={manualLocation}
                  onChange={(e) => setManualLocation(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && manualLocation.trim()) {
                      handleManualLocation();
                    }
                  }}
                  placeholder="Type a location (e.g., Kenyatta Avenue)..."
                  className="w-full px-4 py-3 border-2 border-neutral-300 rounded-xl focus:outline-none focus:border-neutral-900 transition-colors"
                  disabled={geocoding}
                />
              </div>

              {manualLocation.trim() && (
                <Button 
                  onClick={() => handleManualLocation()}
                  className="w-full bg-neutral-900 hover:bg-neutral-800 mb-4"
                  size="lg"
                  disabled={geocoding}
                >
                  {geocoding ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                      Searching...
                    </>
                  ) : (
                    <>
                      <Search className="w-4 h-4 mr-2" />
                      Find Location
                    </>
                  )}
                </Button>
              )}

              <div className="relative mb-4">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-neutral-200"></div>
                </div>
                <div className="relative flex justify-center text-xs">
                  <span className="bg-white px-2 text-neutral-500">Or choose a landmark</span>
                </div>
              </div>

              {/* Landmark selection grid */}
              {loading ? (
                <div className="text-center py-8 text-neutral-500">
                  Loading landmarks...
                </div>
              ) : (
                <div className="max-h-56 overflow-y-auto mb-3 space-y-2 pr-1">
                  {landmarks.map((landmark, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleManualLocation(landmark)}
                      className="w-full text-left px-4 py-3 bg-white border-2 border-neutral-200 rounded-xl hover:border-neutral-900 transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-neutral-900">
                          {landmark.name}
                        </span>
                        <MapPin className="w-4 h-4 text-neutral-400 flex-shrink-0" />
                      </div>
                      <span className="text-xs text-neutral-500">
                        {landmark.coordinates.lat.toFixed(4)}°, {landmark.coordinates.lng.toFixed(4)}°
                      </span>
                    </button>
                  ))}
                </div>
              )}

              <motion.button
                onClick={() => setUseManual(false)}
                className="w-full text-sm text-neutral-600 hover:text-neutral-900 py-2"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Back to location access
              </motion.button>
            </motion.div>
          )}
        </div>
      </motion.div>
    </motion.div>
  );
}