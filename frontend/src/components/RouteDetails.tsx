import { useState, useEffect } from 'react';
import MapboxMap from './MapboxMap';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ArrowLeft, Navigation, Share2, ChevronDown, ChevronUp, TrendingUp, Clock, Route } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import type { RouteData } from '../App';

interface RouteDetailsProps {
  routeData: RouteData;
  onBack: () => void;
  onStartGuidance: () => void;
}

export default function RouteDetails({ routeData, onBack, onStartGuidance }: RouteDetailsProps) {
  const [showDirections, setShowDirections] = useState(false);
  const [fullRouteData, setFullRouteData] = useState<any>(null);

  // Load full route data when the component mounts
  useEffect(() => {
    // If the parent passes full data already, just set it
    setFullRouteData(routeData);
  }, [routeData]);

  const handleShare = () => {
    // Mock share functionality
    alert('Route shared as text');
  };

  return (
    <div className="h-full flex flex-col" style={{ backgroundColor: '#e6e6e6' }}>
      {/* Header */}
      <motion.div 
        className="px-6 py-4 border-b border-neutral-200"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-3">
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
            <h3>Route to {routeData.destination}</h3>
          </div>
        </div>
      </motion.div>

      {/* Map Preview */}
      <div className="relative h-64">
        {fullRouteData && (
          <MapboxMap
            showRiskLayer={false}
            routeData={fullRouteData}
            onWaypointClick={(step) => {
              console.log('Waypoint clicked:', step);
              // Optional: scroll to the relevant direction step
            }}
          />
        )}
      </div>

      {/* Route Info Card */}
      <div className="px-6 py-4 bg-neutral-50 border-b border-neutral-200">
        <div className="flex items-center gap-4 mb-3">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-neutral-600 text-sm">Safety Score</span>
              <Badge className={
                routeData.riskLevel === 'low'
                  ? 'bg-green-100 text-green-800 border-green-300'
                  : routeData.riskLevel === 'high'
                  ? 'bg-red-100 text-red-800 border-red-300'
                  : 'bg-amber-100 text-amber-800 border-amber-300'
              }>
                {routeData.safetyScore}/100
              </Badge>
            </div>
            <div className="h-2 bg-neutral-200 rounded-full overflow-hidden">
              <div
                className={`h-full ${
                  routeData.riskLevel === 'low'
                    ? 'bg-green-500'
                    : routeData.riskLevel === 'high'
                    ? 'bg-red-500'
                    : 'bg-amber-500'
                }`}
                style={{ width: `${routeData.safetyScore}%` }}
              />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-neutral-600" strokeWidth={1.5} />
            <div>
              <p className="text-xs text-neutral-500">ETA</p>
              <p className="text-neutral-900">{routeData.eta} min</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Route className="w-4 h-4 text-neutral-600" strokeWidth={1.5} />
            <div>
              <p className="text-xs text-neutral-500">Distance</p>
              <p className="text-neutral-900">{routeData.distance} km</p>
            </div>
          </div>
        </div>
      </div>

      {/* Directions */}
      <div className="flex-1 overflow-y-auto">
        <motion.button
          onClick={() => setShowDirections(!showDirections)}
          className="w-full px-6 py-4 flex items-center justify-between hover:bg-neutral-50 transition-colors border-b border-neutral-200"
          whileTap={{ scale: 0.99 }}
        >
          <span className="text-neutral-900">Turn-by-turn directions</span>
          {showDirections ? (
            <ChevronUp className="w-5 h-5 text-neutral-600" strokeWidth={2} />
          ) : (
            <ChevronDown className="w-5 h-5 text-neutral-600" strokeWidth={2} />
          )}
        </motion.button>

        <AnimatePresence>
          {showDirections && fullRouteData?.directions && (
            <motion.div 
              className="px-6 py-4 space-y-3"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
            >
              {fullRouteData.directions.map((step: any, idx: number) => {
                // Determine if step is safe based on edge risk
                const isSafe = fullRouteData.metadata?.edge_risks?.[idx] 
                  ? fullRouteData.metadata.edge_risks[idx] < 0.1 
                  : true;

                return (
                  <div key={idx} className="flex gap-3">
                    <div className="flex flex-col items-center">
                      <div className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-sm ${
                        isSafe
                          ? 'border-green-400 text-green-600'
                          : 'border-amber-400 text-amber-600'
                      }`}>
                        {idx + 1}
                      </div>
                      {idx < fullRouteData.directions.length - 1 && (
                        <div className="w-px h-8 bg-neutral-200 my-1" />
                      )}
                    </div>
                    <div className="flex-1 pb-4">
                      <p className="text-neutral-900 mb-1">{step.instruction}</p>
                      <p className="text-sm text-neutral-500">
                        {step.distance_m > 0 ? `${step.distance_m}m` : '—'}
                      </p>
                    </div>
                  </div>
                );
              })}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Bottom Actions */}
      <div className="px-6 py-4 border-t border-neutral-200 space-y-3">
        <Button
          onClick={onStartGuidance}
          className="w-full bg-neutral-900 hover:bg-neutral-800"
          size="lg"
          asChild
        >
          <motion.button whileTap={{ scale: 0.98 }}>
            <Navigation className="w-5 h-5 mr-2" strokeWidth={2} />
            Start Guidance
          </motion.button>
        </Button>
        <Button
          onClick={handleShare}
          variant="outline"
          className="w-full border-2"
          size="lg"
          asChild
        >
          <motion.button whileTap={{ scale: 0.98 }}>
            <Share2 className="w-5 h-5 mr-2" strokeWidth={2} />
            Share Route (text only)
          </motion.button>
        </Button>
      </div>
    </div>
  );
}
