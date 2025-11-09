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
  const API_BASE_URL = import.meta.env.VITE_API_URL;

  const navigateTo = (screen: Screen) => {
    console.log('[App] Navigating to', screen);
    setCurrentScreen(screen);
  };


  // When location permission granted
  const handleLocationGranted = async (location: string, coords: { lat: number; lng: number }) => {
    console.log('[App] Location granted:', location, coords);
    
    setUserLocation(location);
    setUserCoords(coords);
    setShowLocationModal(false);

    toast.loading('Finding your position on the map...', { id: 'find-node' });
    
    try {
      // Find nearest node on the road network
      const response = await fetch(
        `${API_BASE_URL}/nearest-node?lat=${coords.lat}&lng=${coords.lng}`
      );

      if (response.ok) {
        const data = await response.json();
        console.log('[App] Nearest node found:', data.node_id);
        setUserNode(data.node_id);
        toast.success(`Location set: ${location}`, { 
          id: 'find-node',
          description: `Within ${Math.round(data.distance_m)}m of road network`
        });
      } else {
        const error = await response.json();
        throw new Error(error.error || 'Could not find nearest node');
      }
    } catch (error) {
      console.error('Failed to find nearest node:', error);
      toast.error('Location set with limited accuracy', { 
        id: 'find-node',
        description: 'Using approximate position'
      });
      // Fallback to Railway Station as default
      setUserNode('13134429075');
    }

    navigateTo('home-map');
  };

  // Map of landmark destinations to OSM node IDs
  const getNodeIdFromDestination = (destination: string): string => {
    // This will be dynamically fetched from backend based on correct coordinates
    // For now, we'll query the backend for the nearest node
    return '13134429075'; // fallback
  };

  const handleComputeRoute = async (destinationData: RouteData) => {
    if (!userNode) {
      toast.error('User location not set', {
        description: 'Please set your location first'
      });
      return;
    }

    try {
      const startNode = userNode;
      
      // For landmarks, we need to get the node ID from the backend
      // by querying with the landmark coordinates
      let goalNode = '13134429075'; // default fallback
      
      try {
        const landmarksResponse = await fetch(`${API_BASE_URL}/landmarks`);
        if (landmarksResponse.ok) {
          const landmarksData = await landmarksResponse.json();
          const landmark = landmarksData.landmarks.find(
            (l: any) => l.name === destinationData.destination
          );
          if (landmark && landmark.node_id) {
            goalNode = landmark.node_id;
          }
        }
      } catch (error) {
        console.warn('Failed to fetch landmark node IDs, using fallback');
      }

      const riskPreference =
        destinationData.riskLevel === 'low'
          ? 10.0
          : destinationData.riskLevel === 'high'
          ? 1.0
          : 5.0;

      console.log(`[App] Computing route: start=${startNode}, goal=${goalNode}, risk_weight=${riskPreference}`);
      toast.loading('Computing safe route...', { id: 'route-loading' });

      const response = await fetch(
        `${API_BASE_URL}/route?start=${startNode}&goal=${goalNode}&algorithm=astar&lambda_risk=${riskPreference}`
      );

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || `Backend returned ${response.status}`);
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
      toast.error('Failed to compute route', { 
        id: 'route-loading',
        description: error instanceof Error ? error.message : 'Please try again'
      });
    }
  };

  const handleReportSuccess = () => {
    console.log('[App] Report submitted successfully, refreshing map');
    setMapRefreshTrigger((prev) => prev + 1);
    toast.success('Map updated with your report');
  };

  const handleAlertClick = (lat: number, lng: number) => {
    console.log('[App] Alert clicked, navigating to:', lat, lng);
    toast.info('Viewing alert location on map', {
      description: `${lat.toFixed(4)}, ${lng.toFixed(4)}`
    });
    navigateTo('home-map');
  };

  // Handle interactive map click for setting location
  const handleMapLocationSelect = async (lat: number, lng: number) => {
    console.log('[App] Map location selected:', { lat, lng });
    
    // Check bounds
    if (lat < -1.295 || lat > -1.280 || lng < 36.810 || lng > 36.835) {
      toast.error('Location outside Nairobi CBD bounds');
      return;
    }

    toast.loading('Setting location...', { id: 'map-location' });

    try {
      // Reverse geocode
      const geoResponse = await fetch(
        `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json`,
        { headers: { 'User-Agent': 'ProtestSafetyPlanner/1.0' } }
      );
      
      let locationName = 'Nairobi CBD';
      if (geoResponse.ok) {
        const geoData = await geoResponse.json();
        locationName = geoData.address?.road || geoData.address?.neighbourhood || 'Nairobi CBD';
      }

      // Find nearest node
      const response = await fetch(
        `${API_BASE_URL}/nearest-node?lat=${lat}&lng=${lng}`
      );

      if (response.ok) {
        const data = await response.json();
        setUserLocation(locationName);
        setUserCoords({ lat, lng });
        setUserNode(data.node_id);
        
        toast.success(`Location set: ${locationName}`, { 
          id: 'map-location',
          description: `Snapped to nearest road (${Math.round(data.distance_m)}m)`
        });
      } else {
        throw new Error('Could not find nearest road');
      }
    } catch (error) {
      console.error('Map location selection failed:', error);
      toast.error('Failed to set location', { id: 'map-location' });
    }
  };

  const renderScreen = () => {
    try {
      switch (currentScreen) {
        case 'landing':
          return <LandingPage onContinue={() => setShowLocationModal(true)} />;

        case 'home-map':
          return (
            <HomeMap
              userLocation={userLocation}
              userCoords={userCoords}
              onReport={() => setShowReportModal(true)}
              onFindRoute={() => navigateTo('route-destination')}
              onAlerts={() => navigateTo('alerts')}
              onSettings={() => navigateTo('settings')}
              onMapClick={handleMapLocationSelect}
              key={mapRefreshTrigger}
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
    <div className="min-h-screen bg-white overflow-hidden" style={{ backgroundColor: '#e6e6e6' }}>
      <div className="w-full h-screen overflow-hidden relative" style={{ backgroundColor: '#e6e6e6' }}>
        {/* Debug info */}
        <div className="absolute top-0 left-0 z-50 bg-black text-white text-xs px-2 py-1 opacity-75">
          Screen: {currentScreen} | Node: {userNode || 'none'} | 
          Coords: {userCoords ? `${userCoords.lat.toFixed(3)}, ${userCoords.lng.toFixed(3)}` : 'none'}
        </div>

        {renderScreen()}

        {showLocationModal && (
          <LocationPermissionModal
            onLocationGranted={handleLocationGranted}
            onClose={() => setShowLocationModal(false)}
          />
        )}

        {showReportModal && currentScreen === 'home-map' && userCoords && (
          <QuickReportModal
            onClose={() => setShowReportModal(false)}
            userLocation={userCoords}
            onReportSuccess={handleReportSuccess}
          />
        )}
      </div>

      <Toaster position="top-center" />
    </div>
  );
}