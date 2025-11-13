// App.tsx - FIXED with alert count tracking
import { useState, useEffect, useRef } from 'react';
import LandingPage from './components/LandingPage';
import HomePage from './components/HomePage';
import ActivityPage from './components/ActivityPage';
import BottomNav, { NavScreen } from './components/BottomNav';
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
  | 'home'
  | 'activity'
  | 'map'
  | 'location-permission'
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
  const [hasGrantedLocation, setHasGrantedLocation] = useState(false);
  const [alertCount, setAlertCount] = useState(0);
  const homeMapRef = useRef<any>(null);

  const API_BASE_URL = import.meta.env.VITE_API_URL;

  const navigateTo = (screen: Screen) => {
    console.log('[App] Navigating to', screen);
    setCurrentScreen(screen);
  };

  // FETCH ALERT COUNT
  const fetchAlertCount = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/reports/active`);
      if (response.ok) {
        const data = await response.json();
        const count = data.reports?.length || 0;
        setAlertCount(count);
        console.log('[App] Alert count updated:', count);
      }
    } catch (error) {
      console.error('[App] Failed to fetch alert count:', error);
    }
  };

  // POLL FOR ALERTS
  useEffect(() => {
    // Initial fetch
    fetchAlertCount();

    // Poll every 15 seconds
    const interval = setInterval(fetchAlertCount, 15000);

    return () => clearInterval(interval);
  }, []);

  // LOCATION HANDLING
  const handleLocationGranted = async (location: string, coords: { lat: number; lng: number }) => {
    console.log('[App] Location granted:', location, coords);

    setUserLocation(location);
    setUserCoords(coords);
    setHasGrantedLocation(true);
    setShowLocationModal(false);

    toast.loading('Finding your position on the map...', { id: 'find-node' });

    try {
      const response = await fetch(
        `${API_BASE_URL}/nearest-node?lat=${coords.lat}&lng=${coords.lng}`
      );

      if (response.ok) {
        const data = await response.json();
        console.log('[App] Nearest node found:', data.node_id);
        setUserNode(data.node_id);
        toast.success(`Location set: ${location}`, {
          id: 'find-node',
          description: `Within ${Math.round(data.distance_m)}m of road network`,
        });
      } else {
        const error = await response.json();
        throw new Error(error.error || 'Could not find nearest node');
      }
    } catch (error) {
      console.error('Failed to find nearest node:', error);
      toast.error('Location set with limited accuracy', {
        id: 'find-node',
        description: 'Using approximate position',
      });
      setUserNode('13134429075'); // fallback
    }

    navigateTo('map');
  };

  // ROUTE COMPUTATION
  const handleComputeRoute = async (destinationData: RouteData) => {
    if (!userNode) {
      toast.error('User location not set', {
        description: 'Please set your location first',
      });
      return;
    }

    try {
      const startNode = userNode;
      let goalNode = '13134429075'; // fallback

      console.log('[App] Looking up destination:', destinationData.destination);

      try {
        const landmarksResponse = await fetch(`${API_BASE_URL}/landmarks`);
        if (landmarksResponse.ok) {
          const landmarksData = await landmarksResponse.json();
          const landmark = landmarksData.landmarks.find(
            (l: any) => l.name.toLowerCase() === destinationData.destination.toLowerCase()
          );

          if (landmark && landmark.node_id) {
            goalNode = landmark.node_id;
            console.log('[App] ✓ Found landmark node:', goalNode);
          } else {
            console.log('[App] Not a landmark, geocoding...');
            const bbox = '36.810,-1.295,36.835,-1.280';
            const geocodeResponse = await fetch(
              `https://nominatim.openstreetmap.org/search?` +
                `q=${encodeURIComponent(destinationData.destination)}&format=json&limit=1&bounded=1&viewbox=${bbox}&countrycodes=ke`,
              { headers: { 'User-Agent': 'ProtestSafetyPlanner/1.0' } }
            );

            if (geocodeResponse.ok) {
              const geocodeData = await geocodeResponse.json();
              if (geocodeData.length > 0) {
                const destLat = parseFloat(geocodeData[0].lat);
                const destLng = parseFloat(geocodeData[0].lon);

                if (
                  destLat >= -1.295 &&
                  destLat <= -1.280 &&
                  destLng >= 36.810 &&
                  destLng <= 36.835
                ) {
                  console.log('[App] ✓ Geocoded to:', { lat: destLat, lng: destLng });
                  const nodeResponse = await fetch(
                    `${API_BASE_URL}/nearest-node?lat=${destLat}&lng=${destLng}`
                  );

                  if (nodeResponse.ok) {
                    const nodeData = await nodeResponse.json();
                    goalNode = nodeData.node_id;
                    console.log(
                      '[App] ✓ Nearest node:',
                      goalNode,
                      `(${nodeData.distance_m}m away)`
                    );
                  } else {
                    throw new Error('Could not find nearest node');
                  }
                } else {
                  throw new Error('Destination outside Nairobi CBD bounds');
                }
              } else {
                throw new Error('No results found for destination');
              }
            }
          }
        }
      } catch (error) {
        console.error('Destination lookup failed:', error);
        toast.warning('Using approximate destination', {
          description:
            error instanceof Error ? error.message : 'Could not find exact location',
        });
      }

      let algorithm = 'astar';
      let lambda_risk = 10.0;

      if (destinationData.riskLevel === 'low') {
        algorithm = 'dijkstra';
        lambda_risk = 20.0;
      } else if (destinationData.riskLevel === 'high') {
        algorithm = 'astar';
        lambda_risk = 1.0;
      }

      console.log(`[App] Route params: start=${startNode}, goal=${goalNode}`);
      toast.loading('Computing safe route...', { id: 'route-loading' });

      const response = await fetch(
        `${API_BASE_URL}/route?start=${startNode}&goal=${goalNode}&algorithm=${algorithm}&lambda_risk=${lambda_risk}`
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
      toast.success('Route computed successfully!', {
        id: 'route-loading',
        description: `${algorithm.toUpperCase()} • Safety: ${mergedRoute.safetyScore}%`,
      });
      navigateTo('route-details');
    } catch (error) {
      console.error('Route computation failed:', error);
      toast.error('Failed to compute route', {
        id: 'route-loading',
        description:
          error instanceof Error ? error.message : 'Please try again',
      });
    }
  };

  const handleReportSuccess = () => {
    console.log('[App] Report submitted successfully, refreshing map and alerts');
    setMapRefreshTrigger((prev) => prev + 1);
    
    // Immediately refresh alert count
    fetchAlertCount();
    
    toast.success('Map updated with your report');
  };

  const handleAlertClick = (lat: number, lng: number) => {
    console.log('[App] Alert clicked:', lat, lng);
    toast.info('Viewing alert location on map', {
      description: `${lat.toFixed(4)}, ${lng.toFixed(4)}`,
    });
    navigateTo('map');
  };

  const handleMapLocationSelect = async (lat: number, lng: number) => {
    console.log('[App] Map location selected:', { lat, lng });

    if (lat < -1.295 || lat > -1.280 || lng < 36.810 || lng > 36.835) {
      toast.error('Location outside Nairobi CBD bounds');
      return;
    }

    toast.loading('Setting location...', { id: 'map-location' });

    try {
      const geoResponse = await fetch(
        `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json`,
        { headers: { 'User-Agent': 'ProtestSafetyPlanner/1.0' } }
      );

      let locationName = 'Nairobi CBD';
      if (geoResponse.ok) {
        const geoData = await geoResponse.json();
        locationName =
          geoData.address?.road ||
          geoData.address?.neighbourhood ||
          'Nairobi CBD';
      }

      const response = await fetch(`${API_BASE_URL}/nearest-node?lat=${lat}&lng=${lng}`);

      if (response.ok) {
        const data = await response.json();
        setUserLocation(locationName);
        setUserCoords({ lat, lng });
        setUserNode(data.node_id);

        toast.success(`Location set: ${locationName}`, {
          id: 'map-location',
          description: `Snapped to nearest road (${Math.round(data.distance_m)}m)`,
        });
      } else {
        throw new Error('Could not find nearest road');
      }
    } catch (error) {
      console.error('Map location selection failed:', error);
      toast.error('Failed to set location', { id: 'map-location' });
    }
  };

  const handleBottomNavigation = (screen: NavScreen) => {
    if (screen === 'map' && !hasGrantedLocation) {
      setShowLocationModal(true);
      return;
    }
    navigateTo(screen as Screen);
  };

  const shouldShowBottomNav = ['home', 'activity', 'map', 'alerts', 'settings'].includes(currentScreen);

  // SCREEN RENDERING
  const renderScreen = () => {
    switch (currentScreen) {
      case 'landing':
        return <LandingPage onContinue={() => navigateTo('home')} />;
      case 'home':
        return <HomePage />;
      case 'activity':
        return <ActivityPage />;
      case 'map':
        return (
          hasGrantedLocation && (
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
          )
        );
      case 'route-destination':
        return (
          <RouteDestination
            onBack={() => navigateTo('map')}
            onComputeRoute={handleComputeRoute}
          />
        );
      case 'route-details':
        return (
          currentRoute && (
            <RouteDetails
              routeData={currentRoute}
              onBack={() => navigateTo('route-destination')}
              onStartGuidance={() => navigateTo('live-guidance')}
            />
          )
        );
      case 'live-guidance':
        return (
          currentRoute && (
            <LiveGuidance
              routeData={currentRoute}
              onReroute={() => navigateTo('route-destination')}
              onComplete={() => navigateTo('map')}
            />
          )
        );
      case 'alerts':
        return <AlertsFeed onAlertClick={handleAlertClick} />;
      case 'settings':
        return (
          <SettingsPage
            onBack={() => navigateTo('home')}
            onEthics={() => navigateTo('ethics')}
            onHelp={() => navigateTo('help')}
          />
        );
      case 'ethics':
        return <EthicsPage onBack={() => navigateTo('settings')} />;
      case 'help':
        return <HelpPage onBack={() => navigateTo('settings')} />;
      default:
        return <div className="p-4">Unknown screen: {currentScreen}</div>;
    }
  };

  return (
    <div className="min-h-screen overflow-hidden" style={{ backgroundColor: '#e6e6e6' }}>
      <div className="w-full h-screen overflow-hidden relative" style={{ backgroundColor: '#e6e6e6' }}>
        {renderScreen()}

        {showLocationModal && (
          <LocationPermissionModal
            onLocationGranted={handleLocationGranted}
            onClose={() => setShowLocationModal(false)}
          />
        )}

        {showReportModal && currentScreen === 'map' && userCoords && (
          <QuickReportModal
            onClose={() => setShowReportModal(false)}
            userLocation={userCoords}
            onReportSuccess={handleReportSuccess}
          />
        )}

        {shouldShowBottomNav && (
          <BottomNav 
            currentScreen={currentScreen as NavScreen} 
            onNavigate={handleBottomNavigation}
            alertCount={alertCount}
          />
        )}
      </div>

      <Toaster position="top-center" />
    </div>
  );
}