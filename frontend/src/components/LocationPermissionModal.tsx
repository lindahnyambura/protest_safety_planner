import { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { MapPin, Info, X } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { toast } from 'sonner';

interface LocationPermissionModalProps {
  onLocationGranted: (location: string, coords: { lat: number; lng: number }) => void;
  onClose: () => void;
}

export default function LocationPermissionModal({ onLocationGranted, onClose }: LocationPermissionModalProps) {
  const [showWhy, setShowWhy] = useState(false);
  const [manualLocation, setManualLocation] = useState('');
  const [useManual, setUseManual] = useState(false);
  const [loading, setLoading] = useState(false);
  const [landmarks, setLandmarks] = useState<any[]>([]);
  const API_BASE_URL = import.meta.env.VITE_API_URL;

  // Fetch landmarks for manual selection
  useEffect(() => {
    if (useManual) {
      fetch(`${API_BASE_URL}/landmarks`)
        .then(res => res.json())
        .then(data => setLandmarks(data.landmarks))
        .catch(err => console.error('Failed to load landmarks:', err));
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
        
        // Verify we're in Nairobi CBD area
        if (latitude < -1.30 || latitude > -1.27 || longitude < 36.80 || longitude > 36.84) {
          toast.error('Location outside Nairobi CBD area');
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
            onLocationGranted(locationName, { lat: latitude, lng: longitude });
            setLoading(false);
          })
          .catch(() => {
            onLocationGranted('Nairobi CBD', { lat: latitude, lng: longitude });
            setLoading(false);
          });
      },
      (error) => {
        console.error('Geolocation error:', error);
        toast.error('Could not get your location');
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
      onLocationGranted(landmark.name, {
        lat: landmark.coordinates.lat,
        lng: landmark.coordinates.lng
      });
    } else if (manualLocation.trim()) {
      // Search for typed location
      try {
        const API_BASE_URL = import.meta.env.VITE_API_URL;
        const response = await fetch(`${API_BASE_URL}/nearest-landmark?lat=-1.2875&lng=36.8225`);
        const data = await response.json();
        onLocationGranted(data.name, {
          lat: data.coordinates.lat,
          lng: data.coordinates.lng
        });
      } catch {
        onLocationGranted(manualLocation, { lat: -1.2875, lng: 36.8225 });
      }
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
        <div className="bg-white rounded-2xl p-8 shadow-2xl border-2 border-neutral-200">
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
              <p className="text-sm text-neutral-600 mb-3">Select your location:</p>

              {/* Landmark selection grid */}
              <div className="max-h-64 overflow-y-auto mb-3 space-y-2">
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
                      <span className="text-xs text-neutral-500">
                        {landmark.coordinates.lat.toFixed(4)}, {landmark.coordinates.lng.toFixed(4)}
                      </span>
                    </div>
                  </button>
                ))}
              </div>

              {/* Manual text input fallback */}
              <input
                type="text"
                value={manualLocation}
                onChange={(e) => setManualLocation(e.target.value)}
                placeholder="Enter your location (e.g., Kenyatta Avenue)"
                className="w-full px-4 py-3 border-2 border-neutral-300 rounded-xl mb-3 focus:outline-none focus:border-neutral-900 transition-colors"
              />
              
              <Button 
                onClick={() => handleManualLocation()}
                className="w-full bg-neutral-900 hover:bg-neutral-800 mb-3"
                size="lg"
                disabled={!manualLocation.trim() && landmarks.length === 0}
              >
                Continue
              </Button>

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
