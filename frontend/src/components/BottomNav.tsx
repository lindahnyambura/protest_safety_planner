import { Home, MapPin, Bell, Settings, Map } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

export type NavScreen = 'home' | 'activity' | 'map' | 'alerts' | 'settings';

interface BottomNavProps {
  currentScreen: NavScreen;
  onNavigate: (screen: NavScreen) => void;
  alertCount?: number;
}

export default function BottomNav({ currentScreen, onNavigate, alertCount = 0 }: BottomNavProps) {
  const navItems = [
    { id: 'home' as NavScreen, icon: Home, label: 'Home' },
    { id: 'activity' as NavScreen, icon: MapPin, label: 'Activity' },
    { id: 'map' as NavScreen, icon: Map, label: 'Map', isCenter: true },
    { id: 'alerts' as NavScreen, icon: Bell, label: 'Alerts' },
    { id: 'settings' as NavScreen, icon: Settings, label: 'Settings' },
  ];

  return (
    <div
      className="fixed bottom-0 left-0 right-0 border-t-2 border-neutral-900 shadow-lg z-50"
      style={{
        backgroundColor: '#e6e6e6',
        paddingTop: '0.5rem',
        paddingBottom: '0.75rem',
        overflow: 'visible',
      }}
    >
      <div
        className="
          flex items-center justify-around 
          w-full max-w-md mx-auto 
          px-2 sm:px-3 
          overflow-visible
        "
      >
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentScreen === item.id;
          const isCenter = item.isCenter;
          const isAlerts = item.id === 'alerts';
          const hasAlerts = alertCount > 0;

          if (isCenter) {
            return (
              <motion.button
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className="
                  flex flex-col items-center gap-1 relative 
                  px-6 py-2 rounded-2xl transition-all 
                  bg-neutral-900 flex-shrink-0
                "
                whileTap={{ scale: 0.92 }}
                whileHover={{ scale: 1.1 }}
                style={{
                  boxShadow: isActive
                    ? '0 6px 16px rgba(0,0,0,0.25)'
                    : '0 4px 12px rgba(0,0,0,0.15)',
                }}
              >
                <Icon className="w-7 h-7 text-white" strokeWidth={2.5} />
                <span className="text-xs text-white font-semibold">
                  {item.label}
                </span>
              </motion.button>
            );
          }

          return (
            <motion.button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`
                flex flex-col items-center gap-1 relative 
                px-4 py-2 rounded-2xl transition-colors 
                flex-shrink-0
                ${isActive ? 'bg-neutral-900' : ''}
              `}
              whileTap={{ scale: 0.92 }}
              whileHover={{ scale: 1.05 }}
            >
              <div className="relative flex items-center justify-center">
                <Icon
                  className={`w-6 h-6 ${
                    isActive ? 'text-white' : 'text-neutral-600'
                  }`}
                  strokeWidth={2}
                />
                
                {/* SLEEK ALERT BADGE WITH ANIMATIONS */}
                {isAlerts && hasAlerts && (
                  <AnimatePresence>
                    <motion.div
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1}}
                      exit={{ scale: 0, opacity: 0 }}
                      transition={{ 
                        type: 'spring', 
                        stiffness: 500, 
                        damping: 25, 
                      }}
                      className="absolute -top-2 -right-2"
                    >
                      {/* Badge with gradient and border */}
                      <motion.div
                        whileHover={{ scale: 1.15 }}
                        animate={{
                          scale: [1, 1.15, 1],
                          boxShadow: [
                            '0 0 0 0 rgba(0,0,0,0.5)',
                            '0 0 0 6px rgba(0,0,0,0.25)',
                            '0 0 0 0 rgba(0,0,0,0)',
                          ],
                        }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          ease: 'easeInOut',
                        }}
                        className="
                          relative flex items-center justify-center
                          w-5 h-5
                          rounded-full
                          bg-black
                          border border-neutral-800
                          shadow-[0_2px_8px_rgba(0,0,0,0.5)]
                          z-10
                          overflow-hidden
                        "
                      >
                        {/* Count text */}
                        <span 
                          className="
                            text-[10px] font-semibold
                            text-white
                            z-20
                          "
                        >
                          {alertCount > 99 ? '99+' : alertCount}
                        </span>
                        
                        {/* Shine effect that periodically sweeps across */}
                        <motion.div
                          animate={{
                            x: ['-50%', '150%'],
                          }}
                          transition={{
                            duration: 2.5,
                            repeat: Infinity,
                            repeatDelay: 1.5,
                            ease: 'easeInOut',
                          }}
                          className="
                            absolute top-0 bottom-0
                            w-[40%] 
                            bg-gradient-to-r from-transparent via-white/10 to-transparent
                            z-10
                            pointer-events-none
                          "
                          style={{
                            transform: "rotate(20deg)",
                          }}
                        />
                      </motion.div>
                    </motion.div>
                  </AnimatePresence>
                )}
              </div>
              
              <span
                className={`text-xs ${
                  isActive ? 'text-white' : 'text-neutral-600'
                }`}
              >
                {item.label}
              </span>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}