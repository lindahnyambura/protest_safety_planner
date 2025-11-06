import { useState, useEffect } from 'react';
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
  const [currentInstruction, setCurrentInstruction] = useState('Head west on Kenyatta Ave');
  const [distanceToNext, setDistanceToNext] = useState(120);
  const [showHazardAlert, setShowHazardAlert] = useState(false);

  useEffect(() => {
    // Simulate progress
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => onComplete(), 1000);
          return 100;
        }
        return prev + 2;
      });

      setDistanceToNext(prev => Math.max(0, prev - 5));
    }, 500);

    // Simulate hazard detection after 3 seconds
    const hazardTimer = setTimeout(() => {
      setShowHazardAlert(true);
    }, 3000);

    return () => {
      clearInterval(interval);
      clearTimeout(hazardTimer);
    };
  }, [onComplete]);

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Map View */}
      <div className="relative flex-1 bg-neutral-100">
        {/* Mock navigation map */}
        <div className="absolute inset-0 bg-gradient-to-br from-neutral-100 to-neutral-200">
          {/* Grid */}
          <div className="absolute inset-0 opacity-10">
            {[...Array(20)].map((_, i) => (
              <div key={`h-${i}`} className="absolute w-full h-px bg-neutral-400" style={{ top: `${i * 5}%` }} />
            ))}
            {[...Array(20)].map((_, i) => (
              <div key={`v-${i}`} className="absolute h-full w-px bg-neutral-400" style={{ left: `${i * 5}%` }} />
            ))}
          </div>

          {/* Active route with animation */}
          <svg className="absolute inset-0 w-full h-full">
            <motion.path
              d="M 200 500 L 200 300 L 250 300"
              stroke={routeData.riskLevel === 'low' ? '#16a34a' : '#f59e0b'}
              strokeWidth="6"
              fill="none"
              strokeLinecap="round"
              opacity="0.8"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1.5, ease: 'easeInOut' }}
            />
          </svg>

          {/* User position */}
          <motion.div 
            className="absolute transition-all duration-500"
            style={{ 
              left: '200px', 
              top: `${500 - progress * 2}px`,
              transform: 'translate(-50%, -50%)'
            }}
          >
            <div className="relative">
              <div className="w-6 h-6 bg-blue-600 rounded-full border-4 border-white shadow-lg">
                <div className="absolute inset-0 flex items-center justify-center">
                  <Navigation className="w-3 h-3 text-white" strokeWidth={2} />
                </div>
              </div>
              <motion.div 
                className="absolute inset-0 w-6 h-6 bg-blue-600 rounded-full"
                animate={{
                  scale: [1, 2, 1],
                  opacity: [0.5, 0, 0.5],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: 'easeOut',
                }}
              />
            </div>
          </motion.div>
        </div>

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
            <p className="text-xs text-neutral-400 mb-1">IN {distanceToNext} M</p>
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
              <span className="text-white">{Math.max(0, routeData.eta - Math.floor(progress / 10))} min</span>
            </div>
            <div>
              <span className="text-neutral-400">Distance: </span>
              <span className="text-white">{(routeData.distance * (1 - progress / 100)).toFixed(1)} km</span>
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
