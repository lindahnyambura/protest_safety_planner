import { useState, useEffect } from 'react';
import MapboxMap from './MapboxMap';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { AlertTriangle, Navigation, X } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import type { RouteData } from '../App';

interface LiveGuidanceProps {
  routeData: RouteData;
  onReroute: () => void;
  onComplete: () => void;
}

export default function LiveGuidance({ routeData, onReroute, onComplete }: LiveGuidanceProps) {
  const [progress, setProgress] = useState(0);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [showHazardAlert, setShowHazardAlert] = useState(false);

  // Get current instruction from real route data
  const currentInstruction = routeData.directions?.[currentStepIndex]?.instruction || 'Follow the route';
  const distanceToNext = routeData.directions?.[currentStepIndex]?.distance_m || 0;

  useEffect(() => {
    // Simulate progress
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => onComplete(), 1000);
          return 100;
        }
        // Move to next step when reaching milestones
        const newProgress = prev + 2;
        const totalSteps = routeData.directions?.length || 1;
        const stepProgress = Math.floor((newProgress / 100) * totalSteps);
        setCurrentStepIndex(Math.min(stepProgress, totalSteps - 1));

        return newProgress;
      });
    }, 500);

    // Simulate hazard detection
    const hazardTimer = setTimeout(() => {
      setShowHazardAlert(true);
    }, 3000);

    return () => {
      clearInterval(interval);
      clearTimeout(hazardTimer);
    };
  }, [onComplete]);

  return (
    <div className="h-full flex flex-col" style={{ backgroundColor: '#e6e6e6' }}>
      {/* Map View */}
      <div className="relative flex-1">
        <MapboxMap
          showRiskLayer={true}
          routeData={routeData}
          userLocation={routeData.geometry_latlng?.[currentStepIndex]}
        />
        
        {/* Hazard Alert Modal */}
        <AnimatePresence>
          {showHazardAlert && (
            <motion.div 
              className="absolute inset-x-4 top-4 bg-white border-3 border-neutral-900 rounded-2xl p-4 shadow-2xl z-10"
              initial={{ y: -100, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -100, opacity: 0 }}
              transition={{ type: 'spring', stiffness: 300, damping: 25 }}
            >
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-xl bg-neutral-900 flex items-center justify-center flex-shrink-0">
                  <AlertTriangle className="w-5 h-5 text-white" strokeWidth={2} />
                </div>
                <div className="flex-1">
                  <h4 className="text-neutral-900 mb-1">New hazard detected ahead</h4>
                  <p className="text-sm text-neutral-600">Crowd reported on Moi Ave (200m ahead)</p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowHazardAlert(false)}
                  className="flex-shrink-0 rounded-full"
                  asChild
                >
                  <motion.button whileTap={{ scale: 0.9 }}>
                    <X className="w-4 h-4" strokeWidth={2} />
                  </motion.button>
                </Button>
              </div>
              <Button
                onClick={onReroute}
                variant="outline"
                size="sm"
                className="w-full mt-3 border-2 border-neutral-900 text-neutral-900 hover:bg-neutral-900 hover:text-white"
                asChild
              >
                <motion.button whileTap={{ scale: 0.98 }}>
                  Re-route
                </motion.button>
              </Button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Navigation Banner */}
      <div className="bg-neutral-900 text-white px-6 py-6">
        <div className="flex items-start justify-between mb-4">
          <div className="flex-1">
            <p className="text-xs text-neutral-400 mb-1">
              {distanceToNext > 0 ? `IN ${Math.round(distanceToNext)} M` : 'ARRIVING'}
            </p>
            <h3 className="text-white mb-3">{currentInstruction}</h3>
          </div>
          <Navigation className="w-6 h-6 text-white flex-shrink-0 ml-3" strokeWidth={2} />
        </div>

        {/* Progress */}
        <div className="mb-3">
          <Progress value={progress} className="h-2" />
        </div>

        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center gap-4">
            <div>
              <span className="text-neutral-400">ETA: </span>
              <span className="text-white">
                {Math.max(0, routeData.eta - Math.floor(progress / 10))} min
              </span>
            </div>
            <div>
              <span className="text-neutral-400">Distance: </span>
              <span className="text-white">
                {((routeData.distance || 0) * (1 - progress / 100)).toFixed(1)} km
              </span>
            </div>
          </div>
          
          <Badge className={
            routeData.riskLevel === 'low'
              ? 'bg-green-500 text-white border-0'
              : 'bg-amber-500 text-white border-0'
          }>
            {routeData.riskLevel === 'low' ? 'Safe' : 'Caution'}
          </Badge>
        </div>
      </div>

      {/* Bottom Action */}
      <div className="px-6 py-3 bg-white border-t border-neutral-200">
        <Button
          onClick={onComplete}
          variant="outline"
          className="w-full border-2"
          asChild
        >
          <motion.button whileTap={{ scale: 0.98 }}>
            End Navigation
          </motion.button>
        </Button>
      </div>
    </div>
  );
}
