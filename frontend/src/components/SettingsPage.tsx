import { useState } from 'react';
import { Button } from './ui/button';
import { Switch } from './ui/switch';
import { Badge } from './ui/badge';
import { ArrowLeft, ChevronRight, Trash2, FileText, HelpCircle } from 'lucide-react';
import { motion } from 'motion/react';

interface SettingsPageProps {
  onBack: () => void;
  onEthics: () => void;
  onHelp: () => void;
}

export default function SettingsPage({ onBack, onEthics, onHelp }: SettingsPageProps) {
  console.log('[SettingsPage] Component mounted');
  const [settings, setSettings] = useState({
    shareReports: true,
    locationPrecision: true,
    autoReroute: true,
    hapticFeedback: true,
    soundAlerts: false,
  });

  const [reportTTL, setReportTTL] = useState<2 | 5 | 10>(5);

  const toggleSetting = (key: keyof typeof settings) => {
    setSettings(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleDeleteData = () => {
    if (confirm('Delete all local data? This cannot be undone.')) {
      alert('Local data deleted');
    }
  };

  return (
    <div className="h-full flex flex-col overflow-y-auto" style={{ backgroundColor: '#e6e6e6' }}>
      {/* Header */}
      <motion.div 
        className="px-6 py-4 border-b border-neutral-200 bg-white sticky top-0 z-10"
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
          <h2>Settings & Privacy</h2>
        </div>
      </motion.div>

      {/* Content */}
      <div className="flex-1">
        {/* Privacy Settings */}
        <div className="px-6 py-6 border-b border-neutral-200">
          <h3 className="mb-4 text-neutral-900">Privacy</h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex-1 pr-4">
                <p className="text-neutral-900 mb-1">Share Reports</p>
                <p className="text-sm text-neutral-600">
                  Contribute anonymous safety reports to the community
                </p>
              </div>
              <Switch
                checked={settings.shareReports}
                onCheckedChange={() => toggleSetting('shareReports')}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="flex-1 pr-4">
                <p className="text-neutral-900 mb-1">High Location Precision</p>
                <p className="text-sm text-neutral-600">
                  Use precise location for better route accuracy
                </p>
              </div>
              <Switch
                checked={settings.locationPrecision}
                onCheckedChange={() => toggleSetting('locationPrecision')}
              />
            </div>
          </div>
        </div>

        {/* Navigation Settings */}
        <div className="px-6 py-6 border-b border-neutral-200">
          <h3 className="mb-4 text-neutral-900">Navigation</h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex-1 pr-4">
                <p className="text-neutral-900 mb-1">Auto Re-route</p>
                <p className="text-sm text-neutral-600">
                  Automatically find safer routes when hazards appear
                </p>
              </div>
              <Switch
                checked={settings.autoReroute}
                onCheckedChange={() => toggleSetting('autoReroute')}
              />
            </div>
          </div>
        </div>

        {/* Alerts Settings */}
        <div className="px-6 py-6 border-b border-neutral-200">
          <h3 className="mb-4 text-neutral-900">Alerts</h3>
          
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex-1 pr-4">
                <p className="text-neutral-900 mb-1">Haptic Feedback</p>
                <p className="text-sm text-neutral-600">
                  Vibrate on important alerts
                </p>
              </div>
              <Switch
                checked={settings.hapticFeedback}
                onCheckedChange={() => toggleSetting('hapticFeedback')}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="flex-1 pr-4">
                <p className="text-neutral-900 mb-1">Sound Alerts</p>
                <p className="text-sm text-neutral-600">
                  Play sound for critical warnings
                </p>
              </div>
              <Switch
                checked={settings.soundAlerts}
                onCheckedChange={() => toggleSetting('soundAlerts')}
              />
            </div>
          </div>
        </div>

        {/* Data Retention */}
        <div className="px-6 py-6 border-b border-neutral-200">
          <h3 className="mb-4 text-neutral-900">Data Retention</h3>
          
          <div>
            <p className="text-neutral-900 mb-3">Report Expiry Time</p>
            <div className="flex gap-2">
              {([2, 5, 10] as const).map((minutes) => (
                <Badge
                  key={minutes}
                  onClick={() => setReportTTL(minutes)}
                  className={`flex-1 justify-center cursor-pointer py-2 ${
                    reportTTL === minutes
                      ? 'bg-neutral-900 text-white'
                      : 'bg-white text-neutral-600 border-neutral-300'
                  }`}
                >
                  {minutes} min
                </Badge>
              ))}
            </div>
            <p className="text-sm text-neutral-600 mt-3">
              Reports older than this will be automatically deleted
            </p>
          </div>

          <div className="mt-6">
            <Button
              onClick={handleDeleteData}
              variant="outline"
              size="sm"
              className="text-red-600 border-red-300 hover:bg-red-50"
              asChild
            >
              <motion.button whileTap={{ scale: 0.98 }}>
                <Trash2 className="w-4 h-4 mr-2" strokeWidth={2} />
                Delete Local Data
              </motion.button>
            </Button>
          </div>
        </div>

        {/* Information Links */}
        <div className="px-6 py-6 space-y-3">
          <motion.button
            onClick={onEthics}
            className="w-full flex items-center justify-between py-3 hover:bg-neutral-50 rounded-lg px-3 transition-colors"
            whileTap={{ scale: 0.99 }}
          >
            <div className="flex items-center gap-3">
              <FileText className="w-5 h-5 text-neutral-600" strokeWidth={2} />
              <span className="text-neutral-900">Ethics & DPIA</span>
            </div>
            <ChevronRight className="w-5 h-5 text-neutral-400" strokeWidth={2} />
          </motion.button>

          <motion.button
            onClick={onHelp}
            className="w-full flex items-center justify-between py-3 hover:bg-neutral-50 rounded-lg px-3 transition-colors"
            whileTap={{ scale: 0.99 }}
          >
            <div className="flex items-center gap-3">
              <HelpCircle className="w-5 h-5 text-neutral-600" strokeWidth={2} />
              <span className="text-neutral-900">Help & Safety Tips</span>
            </div>
            <ChevronRight className="w-5 h-5 text-neutral-400" strokeWidth={2} />
          </motion.button>
        </div>

        {/* Footer */}
        <div className="px-6 py-8 text-center">
          <p className="text-sm text-neutral-500">SafeNav v1.0.0</p>
          <p className="text-sm text-neutral-500">Open Source • No Tracking</p>
        </div>
      </div>
    </div>
  );
}
