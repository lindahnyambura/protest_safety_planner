import { useState } from "react";
import { Toaster } from "./components/ui/sonner";
import { Landing } from "./components/Landing";
import { Onboarding } from "./components/Onboarding";
import { HomeMap } from "./components/HomeMap";
import { QuickReport } from "./components/QuickReport";
import { RouteDestination } from "./components/RouteDestination";
import { RouteDetails } from "./components/RouteDetails";
import { LiveGuidance } from "./components/LiveGuidance";
import { AlertsFeed } from "./components/AlertsFeed";
import { Settings } from "./components/Settings";
import { Help } from "./components/Help";
import { Ethics } from "./components/Ethics";

export type Screen = 
  | "landing"
  | "onboarding" 
  | "home" 
  | "quick-report" 
  | "route-destination" 
  | "route-details" 
  | "live-guidance" 
  | "alerts" 
  | "settings" 
  | "help"
  | "ethics";

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<Screen>("landing");
  const [navigationStack, setNavigationStack] = useState<Screen[]>([]);

  const navigateTo = (screen: Screen, data?: any) => {
    setNavigationStack((prev) => [...prev, currentScreen]);
    setCurrentScreen(screen);
  };

  const navigateBack = () => {
    if (navigationStack.length > 0) {
      const previousScreen = navigationStack[navigationStack.length - 1];
      setNavigationStack((prev) => prev.slice(0, -1));
      setCurrentScreen(previousScreen);
    } else {
      // Default back to home if no stack
      setCurrentScreen("home");
    }
  };

  const handleLandingComplete = () => {
    setCurrentScreen("home");
  };

  const handleOnboardingComplete = () => {
    setCurrentScreen("home");
  };

  const closeQuickReport = () => {
    setCurrentScreen("home");
  };

  const backToLanding = () => {
    setCurrentScreen("landing");
    setNavigationStack([]);
  };

  return (
    <div className="min-h-screen bg-[#FDF8F0]">
      {/* Render appropriate screen */}
      {currentScreen === "landing" && (
        <Landing onComplete={handleLandingComplete} />
      )}
      
      {currentScreen === "onboarding" && (
        <Onboarding onComplete={handleOnboardingComplete} />
      )}
      
      {currentScreen === "home" && (
        <HomeMap onNavigate={navigateTo} onBack={backToLanding} />
      )}
      
      {currentScreen === "quick-report" && (
        <QuickReport onClose={closeQuickReport} />
      )}
      
      {currentScreen === "route-destination" && (
        <RouteDestination 
          onNavigate={navigateTo} 
          onBack={navigateBack}
        />
      )}
      
      {currentScreen === "route-details" && (
        <RouteDetails 
          onNavigate={navigateTo} 
          onBack={navigateBack}
        />
      )}
      
      {currentScreen === "live-guidance" && (
        <LiveGuidance 
          onNavigate={navigateTo} 
          onBack={navigateBack}
        />
      )}
      
      {currentScreen === "alerts" && (
        <AlertsFeed 
          onBack={navigateBack}
          onSelectAlert={(alert) => {
            // Could navigate to home and highlight location
            setCurrentScreen("home");
          }}
        />
      )}
      
      {currentScreen === "settings" && (
        <Settings 
          onNavigate={navigateTo} 
          onBack={navigateBack}
        />
      )}
      
      {currentScreen === "help" && (
        <Help onBack={navigateBack} />
      )}
      
      {currentScreen === "ethics" && (
        <Ethics onBack={navigateBack} />
      )}

      {/* Toast notifications */}
      <Toaster 
        toastOptions={{
          style: {
            background: '#FDF8F0',
            color: '#000',
            border: '2px solid #000',
            borderRadius: '0',
          },
        }}
      />
    </div>
  );
}