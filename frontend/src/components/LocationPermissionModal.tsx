import { useState } from 'react';
import { Button } from './ui/button';
import { MapPin, Info, X } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface LocationPermissionModalProps {
  onLocationGranted: (location: string) => void;
  onClose: () => void;
}

export default function LocationPermissionModal({ onLocationGranted, onClose }: LocationPermissionModalProps) {
  const [showWhy, setShowWhy] = useState(false);
  const [manualLocation, setManualLocation] = useState('');
  const [useManual, setUseManual] = useState(false);

  const handleGrantPermission = () => {
    // Simulate getting user's location
    onLocationGranted('Nairobi CBD, Kenya');
  };

  const handleManualLocation = () => {
    if (manualLocation.trim()) {
      onLocationGranted(manualLocation);
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
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <Button 
                onClick={handleGrantPermission}
                className="w-full bg-neutral-900 hover:bg-neutral-800 mb-3"
                size="lg"
                asChild
              >
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  Allow Location Access
                </motion.button>
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
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
            >
              <input
                type="text"
                value={manualLocation}
                onChange={(e) => setManualLocation(e.target.value)}
                placeholder="Enter your location"
                className="w-full px-4 py-3 border-2 border-neutral-300 rounded-xl mb-3 focus:outline-none focus:border-neutral-900 transition-colors"
              />
              
              <Button 
                onClick={handleManualLocation}
                className="w-full bg-neutral-900 hover:bg-neutral-800 mb-3"
                size="lg"
                disabled={!manualLocation.trim()}
                asChild
              >
                <motion.button
                  whileHover={{ scale: !manualLocation.trim() ? 1 : 1.02 }}
                  whileTap={{ scale: !manualLocation.trim() ? 1 : 0.98 }}
                >
                  Continue
                </motion.button>
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
