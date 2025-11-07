import { useState, useEffect } from 'react';
import LandingPage from './components/LandingPage';
import LocationPermissionModal from './components/LocationPermissionModal';
import HomeMap from './components/HomeMap';
import QuickReportModal from './components/QuickReportModal';
import RouteDestination from './components/RouteDestination';
import RouteDetails from './components/RouteDetails';
import LiveGuidance from './components/LiveGuidance';
import AlertsFeed from './components/AlertsFeed';
import SettingsPage from './components/SettingsPage';
import EthicsPage from './components/EthicsPage';
import HelpPage from './components/HelpPage';
import { Toaster } from './components/ui/sonner';
import { toast } from 'sonner';

export type Screen = 
  | 'landing'
  | 'location-permission'
  | 'home-map'
  | 'route-destination'
  | 'route-details'
  | 'live-guidance'
  | 'alerts'
  | 'settings'
  | 'ethics'
  | 'help';

export interface RouteData {
  destination: string;
  safetyScore: number;
  eta: number;
  distance: number;
  riskLevel: 'low' | 'medium' | 'high';
  // Backend data (merged)
  geometry_latlng?: [number, number][];
  directions?: any[];
  metadata?: any;
  path?: string[];
}

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<Screen>('landing');
  const [showReportModal, setShowReportModal] = useState(false);
  const [showLocationModal, setShowLocationModal] = useState(false);
  const [currentRoute, setCurrentRoute] = useState<RouteData | null>(null);
  const [userLocation, setUserLocation] = useState<string>('');
  const [userCoords, setUserCoords] = useState<{ lat: number; lng: number } | null>(null);
  const [userNode, setUserNode] = useState<string | null>(null);

  const navigateTo = (screen: Screen) => {
    console.log('[App] Navigating to', screen);
    setCurrentScreen(screen);
  };

  // Debug
  useEffect(() => {
    console.log('[App] Current screen:', currentScreen);
  }, [currentScreen]);

  const handleLocationGranted = async (location: string, coords: { lat: number; lng: number }) => {
    setUserLocation(location);
    setUserCoords(coords);
    setShowLocationModal(false);

    // Find nearest OSM node to user's location
    toast.loading('Finding your position...', { id: 'find-node' });
    
    try {
      const response = await fetch(
        `http://localhost:8000/nearest-node?lat=${coords.lat}&lng=${coords.lng}`
      );
      
      if (response.ok) {
        const data = await response.json();
        setUserNode(data.node_id);
        toast.success('Location set successfully!', { id: 'find-node' });
      } else {
        throw new Error('Could not find nearest node');
      }
    } catch (error) {
      console.error('Failed to find nearest node:', error);
      toast.error('Location set with limited accuracy', { id: 'find-node' });
      // Fallback to default node
      setUserNode('12361156623');  // Odeon
    }
    
    navigateTo('home-map');
  };

  // Update destination mapping with real Nairobi landmarks
  const getNodeIdFromDestination = (destination: string): string => {
    const destinationMap: Record<string, string> = {
      // Real Nairobi CBD landmarks with actual OSM nodes
      'Jamia Mosque': '6580961457',      // Near Tom Mboya Street
      'National Archives': '12414258058',   // On Moi Avenue
      'Afya Center': '10873342295',      // Near Tom Mboya
      'GPO (General Post Office)': '12361445752', // Kenyatta Avenue
      'Railway Station': '8584796189',   // Near Uhuru Highway
      'KICC': '13134429074',              // City center landmark
      
      // Fallback for old names
      "Uhuru Park": "12343642875",
      "City Market": "9859577513",
      "Kencom": "12343534285",
      "Bus Station": "10873342299",
      "Teleposta Towers": "5555073936",
      "Times Tower": "10701041875",
      "Odeon": "12361156623",
    };

    return destinationMap[destination] || "9859577513";  // city market
  };

  const handleComputeRoute = async (destinationData: RouteData) => {
    if (!userNode) {
      toast.error('User location not set');
      return;
    }

    try {
      const startNode = userNode;
      const goalNode = getNodeIdFromDestination(destinationData.destination);

      console.log(`[App] Computing route from ${startNode} to ${goalNode}`);
      toast.loading('Computing safe route...', { id: 'route-loading' });

      const response = await fetch(
        `http://localhost:8000/route?start=${startNode}&goal=${goalNode}&algorithm=astar`
      );

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const backendRoute = await response.json();

      // Merge backend data with UI data
      const mergedRoute: RouteData = {
        ...destinationData,
        geometry_latlng: backendRoute.geometry_latlng,
        directions: backendRoute.directions,
        metadata: backendRoute.metadata,
        path: backendRoute.path,
        safetyScore: Math.round(backendRoute.safety_score * 100),
        eta: Math.round(backendRoute.metadata.estimated_time_s / 60),
        distance: parseFloat((backendRoute.metadata.total_distance_m / 1000).toFixed(1)),
        riskLevel: backendRoute.metadata.max_edge_risk > 0.3 
          ? 'high' 
          : backendRoute.metadata.max_edge_risk > 0.1 
          ? 'medium' 
          : 'low'
      };

      setCurrentRoute(mergedRoute);
      toast.success('Route computed successfully!', { id: 'route-loading' });
      setCurrentScreen('route-details');

    } catch (error) {
      console.error('Route computation failed:', error);
      toast.error('Failed to compute route. Please try again.', { id: 'route-loading' });
    }
  };

  return (
    <div className="min-h-screen bg-white overflow-hidden">
      <div className="w-full h-screen bg-white overflow-hidden relative">
        {/* Add debug info at top */}
        <div className="absolute top-0 left-0 z-50 bg-black text-white text-xs px-2 py-1">
          Screen: {currentScreen}
        </div>

        {currentScreen === 'landing' && (
          <LandingPage onContinue={() => setShowLocationModal(true)} />
        )}
        
        {showLocationModal && (
          <LocationPermissionModal 
            onLocationGranted={handleLocationGranted}
            onClose={() => setShowLocationModal(false)}
          />
        )}
        
        {currentScreen === 'home-map' && (
          <HomeMap
            userLocation={userLocation}
            onReport={() => setShowReportModal(true)}
            onFindRoute={() => navigateTo('route-destination')}
            onAlerts={() => navigateTo('alerts')}
            onSettings={() => navigateTo('settings')}
          />
        )}
        
        {currentScreen === 'route-destination' && (
          <RouteDestination
            onBack={() => navigateTo('home-map')}
            onComputeRoute={handleComputeRoute}
          />
        )}
        
        {currentScreen === 'route-details' && currentRoute && (
          <RouteDetails
            routeData={currentRoute}
            onBack={() => navigateTo('route-destination')}
            onStartGuidance={() => navigateTo('live-guidance')}
          />
        )}
        
        {currentScreen === 'live-guidance' && currentRoute && (
          <LiveGuidance
            routeData={currentRoute}
            onReroute={() => navigateTo('route-destination')}
            onComplete={() => navigateTo('home-map')}
          />
        )}
        
        {currentScreen === 'alerts' && (
          <AlertsFeed onBack={() => navigateTo('home-map')} />
        )}
        
        {currentScreen === 'settings' && (
          <SettingsPage
            onBack={() => navigateTo('home-map')}
            onEthics={() => navigateTo('ethics')}
            onHelp={() => navigateTo('help')}
          />
        )}
        
        {currentScreen === 'ethics' && (
          <EthicsPage onBack={() => navigateTo('settings')} />
        )}
        
        {currentScreen === 'help' && (
          <HelpPage onBack={() => navigateTo('settings')} />
        )}

        {showReportModal && currentScreen === 'home-map' && (
          <QuickReportModal onClose={() => setShowReportModal(false)} />
        )}
      </div>
      
      <Toaster />
    </div>
  );
}