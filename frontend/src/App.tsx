import { useState, useEffect, useRef } from 'react';
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
  geometry_latlng?: [number, number][];
  directions?: Array<{
    step: number;
    lat: number;
    lng: number;
    instruction: string;
    street_name: string | null;
    distance_m: number;
  }>;
  metadata?: {
    total_distance_m: number;
    estimated_time_s: number;
    mean_edge_risk: number;
    max_edge_risk: number;
    edge_risks: number[];
  };
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
  const [mapRefreshTrigger, setMapRefreshTrigger] = useState(0);
  const homeMapRef = useRef<any>(null);

  const navigateTo = (screen: Screen) => {
    console.log('[App] Navigating to', screen);
    setCurrentScreen(screen);
  };

  const API_BASE_URL = import.meta.env.VITE_API_URL;

  // When location permission granted
  const handleLocationGranted = async (location: string, coords: { lat: number; lng: number }) => {
    setUserLocation(location);
    setUserCoords(coords);
    setShowLocationModal(false);

    toast.loading('Finding your position...', { id: 'find-node' });
    try {
      const response = await fetch(
        `${API_BASE_URL}/nearest-node?lat=${coords.lat}&lng=${coords.lng}`
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
      setUserNode('13134429075'); // fallback (Odeon) // railway
    }

    navigateTo('home-map');
  };

  // Map of landmark destinations to OSM node IDs
  const getNodeIdFromDestination = (destination: string): string => {
    const destinationMap: Record<string, string> = {
      'Jamia Mosque': '6807551638',
      'National Archives': '12168049898',
      'Afya Center': '10873342299',
      'GPO (General Post Office)': '11895806370',
      'Railway Station': '13134429075',
      // 'KICC': '13134429074',
      'Uhuru Park': '11895806775',
      'City Market': '12364334298',
      'Kencom': '12343534285',
      'Bus Station': '8555798083',
      'Koja': '11895812325',
      'Times Tower': '12361123931',
      'Odeon': '12361156623',
    };
    return destinationMap[destination] || '13134429075'; // default fallback
  };

  const handleComputeRoute = async (destinationData: RouteData) => {
    if (!userNode) {
      toast.error('User location not set');
      return;
    }

    try {
      const startNode = userNode;
      const goalNode = getNodeIdFromDestination(destinationData.destination);

      const riskPreference =
        destinationData.riskLevel === 'low'
          ? 10.0 // safest
          : destinationData.riskLevel === 'high'
          ? 1.0 // fastest
          : 5.0; // balanced

      console.log(`[App] Computing route: start=${startNode}, goal=${goalNode}, risk_weight=${riskPreference}`);
      toast.loading('Computing safe route...', { id: 'route-loading' });

      const response = await fetch(
        `protestsafetyplanner-production.up.railway.app/route?start=${startNode}&goal=${goalNode}&algorithm=astar&lambda_risk=${riskPreference}`
      );

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const backendRoute = await response.json();

      const mergedRoute: RouteData = {
        ...destinationData,
        geometry_latlng: backendRoute.geometry_latlng,
        directions: backendRoute.directions,
        metadata: backendRoute.metadata,
        path: backendRoute.path,
        safetyScore: Math.round(backendRoute.safety_score * 100),
        eta: Math.round(backendRoute.metadata.estimated_time_s / 60),
        distance: parseFloat((backendRoute.metadata.total_distance_m / 1000).toFixed(1)),
        riskLevel:
          backendRoute.metadata.max_edge_risk > 0.3
            ? 'high'
            : backendRoute.metadata.max_edge_risk > 0.1
            ? 'medium'
            : 'low',
      };

      setCurrentRoute(mergedRoute);
      toast.success('Route computed successfully!', { id: 'route-loading' });
      navigateTo('route-details');
    } catch (error) {
      console.error('Route computation failed:', error);
      toast.error('Failed to compute route. Please try again.', { id: 'route-loading' });
    }
  };

  const handleReportSuccess = () => {
    console.log('[App] Report submitted successfully, refreshing map');
    setMapRefreshTrigger((prev) => prev + 1);
  };

  const handleAlertClick = (lat: number, lng: number) => {
    console.log('[App] Alert clicked, navigating to:', lat, lng);
    toast.info('Viewing alert location on map', {
      description: `${lat.toFixed(4)}, ${lng.toFixed(4)}`
    });
    navigateTo('home-map');
  };

  const renderScreen = () => {
    try {
      switch (currentScreen) {
        case 'landing':
          return <LandingPage onContinue={() => setShowLocationModal(true)} />;

        case 'home-map':
          return (
            <HomeMap
              // ref={homeMapRef}
              userLocation={userLocation}
              onReport={() => setShowReportModal(true)}
              onFindRoute={() => navigateTo('route-destination')}
              onAlerts={() => navigateTo('alerts')}
              onSettings={() => navigateTo('settings')}
              key={mapRefreshTrigger} // trigger refresh
            />
          );

        case 'route-destination':
          return (
            <RouteDestination
              onBack={() => navigateTo('home-map')}
              onComputeRoute={handleComputeRoute}
            />
          );

        case 'route-details':
          if (!currentRoute) {
            toast.error('No route data available');
            navigateTo('route-destination');
            return null;
          }
          return (
            <RouteDetails
              routeData={currentRoute}
              onBack={() => navigateTo('route-destination')}
              onStartGuidance={() => navigateTo('live-guidance')}
            />
          );

        case 'live-guidance':
          if (!currentRoute) {
            toast.error('No route data available');
            navigateTo('home-map');
            return null;
          }
          return (
            <LiveGuidance
              routeData={currentRoute}
              onReroute={() => navigateTo('route-destination')}
              onComplete={() => navigateTo('home-map')}
            />
          );

        case 'alerts':
          return <AlertsFeed onBack={() => navigateTo('home-map')} onAlertClick={handleAlertClick} />;

        case 'settings':
          return (
            <SettingsPage
              onBack={() => navigateTo('home-map')}
              onEthics={() => navigateTo('ethics')}
              onHelp={() => navigateTo('help')}
            />
          );

        case 'ethics':
          return <EthicsPage onBack={() => navigateTo('settings')} />;

        case 'help':
          return <HelpPage onBack={() => navigateTo('settings')} />;

        default:
          console.error('[App] Unknown screen:', currentScreen);
          return <div className="p-4">Unknown screen: {currentScreen}</div>;
      }
    } catch (error) {
      console.error('[App] Error rendering screen:', error);
      return (
        <div className="flex items-center justify-center h-screen p-4">
          <div className="text-center">
            <h2 className="text-xl font-bold mb-2">Something went wrong</h2>
            <p className="text-neutral-600 mb-4">
              {error instanceof Error ? error.message : 'Unknown error'}
            </p>
            <button
              onClick={() => navigateTo('home-map')}
              className="px-4 py-2 bg-neutral-900 text-white rounded-lg"
            >
              Return to Home
            </button>
          </div>
        </div>
      );
    }
  };

  return (
    <div className="min-h-screen bg-white overflow-hidden">
      <div className="w-full h-screen bg-white overflow-hidden relative">
        <div className="absolute top-0 left-0 z-50 bg-black text-white text-xs px-2 py-1 opacity-75">
          Screen: {currentScreen} | Reports: {mapRefreshTrigger}
        </div>

        {renderScreen()}

        {showLocationModal && (
          <LocationPermissionModal
            onLocationGranted={handleLocationGranted}
            onClose={() => setShowLocationModal(false)}
          />
        )}

        {showReportModal && currentScreen === 'home-map' && (
          <QuickReportModal
            onClose={() => setShowReportModal(false)}
            userLocation={userCoords ? [userCoords.lat, userCoords.lng] : undefined}
            onReportSuccess={handleReportSuccess}
          />
        )}
      </div>

      <Toaster position="top-center" />
    </div>
  );
}
