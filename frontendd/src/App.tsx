import { useState } from 'react';
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
}

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<Screen>('landing');
  const [showReportModal, setShowReportModal] = useState(false);
  const [showLocationModal, setShowLocationModal] = useState(false);
  const [routeData, setRouteData] = useState<RouteData | null>(null);
  const [userLocation, setUserLocation] = useState<string>('');

  const navigateTo = (screen: Screen) => {
    setCurrentScreen(screen);
  };

  const handleLocationGranted = (location: string) => {
    setUserLocation(location);
    setShowLocationModal(false);
    navigateTo('home-map');
  };

  const handleComputeRoute = (data: RouteData) => {
    setRouteData(data);
    navigateTo('route-details');
  };

  return (
    <div className="min-h-screen bg-white overflow-hidden">
      <div className="w-full h-screen bg-white overflow-hidden relative">
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
        
        {currentScreen === 'route-details' && routeData && (
          <RouteDetails
            routeData={routeData}
            onBack={() => navigateTo('route-destination')}
            onStartGuidance={() => navigateTo('live-guidance')}
          />
        )}
        
        {currentScreen === 'live-guidance' && routeData && (
          <LiveGuidance
            routeData={routeData}
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
