import { useState } from "react";
import { motion } from "motion/react";
import { ArrowLeft, HelpCircle, Trash2, Shield } from "lucide-react";
import { toast } from "sonner";

interface SettingsProps {
  onNavigate: (screen: string) => void;
  onBack: () => void;
}

export function Settings({ onNavigate, onBack }: SettingsProps) {
  const [settings, setSettings] = useState({
    shareReports: true,
    locationPrecision: true,
    autoReroute: true,
    hapticFeedback: false,
    soundAlerts: true,
  });
  
  const [ttl, setTtl] = useState<2 | 5 | 10>(5);

  const toggleSetting = (key: keyof typeof settings) => {
    setSettings((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const handleDeleteData = () => {
    toast("Local data cleared", {
      description: "All cached data has been deleted",
    });
  };

  return (
    <div className="min-h-screen bg-[#FDF8F0] flex flex-col">
      {/* Header */}
      <div className="border-b-2 border-black bg-[#FDF8F0] p-4 flex items-center gap-3">
        <button
          onClick={onBack}
          className="border-2 border-black p-2 hover:bg-[#E8E3D8] transition-colors"
        >
          <ArrowLeft size={20} />
        </button>
        <h1 className="text-lg">SETTINGS & PRIVACY</h1>
      </div>

      <div className="flex-1 overflow-y-auto">
        {/* Settings List */}
        <div className="divide-y-2 divide-black">
          <SettingToggle
            label="Share Reports"
            description="Contribute anonymously to community safety"
            checked={settings.shareReports}
            onChange={() => toggleSetting("shareReports")}
          />
          <SettingToggle
            label="High Location Precision"
            description="Use GPS for more accurate hazard detection"
            checked={settings.locationPrecision}
            onChange={() => toggleSetting("locationPrecision")}
          />
          <SettingToggle
            label="Auto Re-route"
            description="Automatically suggest safer routes when hazards appear"
            checked={settings.autoReroute}
            onChange={() => toggleSetting("autoReroute")}
          />
          <SettingToggle
            label="Haptic Feedback"
            description="Vibrate on important alerts"
            checked={settings.hapticFeedback}
            onChange={() => toggleSetting("hapticFeedback")}
          />
          <SettingToggle
            label="Sound Alerts"
            description="Play audio for critical warnings"
            checked={settings.soundAlerts}
            onChange={() => toggleSetting("soundAlerts")}
          />
        </div>

        {/* Privacy Section */}
        <div className="p-6 space-y-4">
          {/* TTL Selector */}
          <div className="border-2 border-black p-4 bg-[#FDF8F0]">
            <h3 className="text-sm mb-3" style={{ fontWeight: 600 }}>REPORT EXPIRY</h3>
            <div className="flex gap-2 mb-2">
              {([2, 5, 10] as const).map((minutes) => (
                <button
                  key={minutes}
                  onClick={() => setTtl(minutes)}
                  className="flex-1 border-2 border-black p-2 text-sm hover:bg-[#E8E3D8] transition-colors"
                  style={{
                    backgroundColor: ttl === minutes ? "#04771B" : "#FDF8F0",
                    borderWidth: ttl === minutes ? "3px" : "2px",
                  }}
                >
                  {minutes} min
                </button>
              ))}
            </div>
            <p className="text-xs opacity-60">
              Choose how long your data lives before automatic deletion
            </p>
          </div>

          <div className="border-2 border-black p-4 bg-[#E8E3D8]">
            <div className="flex items-center gap-3 mb-3">
              <Shield size={20} />
              <h3 className="text-sm">PRIVACY FIRST</h3>
            </div>
            <p className="text-xs leading-relaxed">
              All processing happens on your device. No accounts, no tracking, no data sold. 
              Reports are anonymized before sharing.
            </p>
          </div>

          <button
            onClick={handleDeleteData}
            className="w-full border-2 border-black bg-[#FDF8F0] p-4 flex items-center gap-3 hover:bg-[#AE1E2A] transition-colors"
          >
            <Trash2 size={20} />
            <div className="flex-1 text-left">
              <p className="text-sm">Delete Local Data</p>
              <p className="text-xs opacity-60">Clear all cached information</p>
            </div>
          </button>

          <button
            onClick={() => onNavigate("help")}
            className="w-full border-2 border-black bg-[#FDF8F0] p-4 flex items-center gap-3 hover:bg-[#E8E3D8] transition-colors"
          >
            <HelpCircle size={20} />
            <div className="flex-1 text-left">
              <p className="text-sm">Help & Safety Tips</p>
              <p className="text-xs opacity-60">Learn how to stay safe</p>
            </div>
          </button>

          <button
            onClick={() => onNavigate("ethics")}
            className="w-full border-2 border-black bg-[#FDF8F0] p-4 flex items-center gap-3 hover:bg-[#E8E3D8] transition-colors"
          >
            <Shield size={20} />
            <div className="flex-1 text-left">
              <p className="text-sm">Ethics & DPIA</p>
              <p className="text-xs opacity-60">Our data protection impact assessment</p>
            </div>
          </button>
        </div>
      </div>

      {/* App Version */}
      <div className="border-t-2 border-black bg-[#E8E3D8] p-4 text-center">
        <p className="text-xs opacity-60">SafeNav v1.0.0 • No tracking • Open source</p>
      </div>
    </div>
  );
}

interface SettingToggleProps {
  label: string;
  description: string;
  checked: boolean;
  onChange: () => void;
}

function SettingToggle({ label, description, checked, onChange }: SettingToggleProps) {
  return (
    <div className="p-4 flex items-start gap-4">
      <div className="flex-1">
        <p className="text-sm mb-1">{label}</p>
        <p className="text-xs opacity-60">{description}</p>
      </div>
      <motion.button
        onClick={onChange}
        className="border-2 border-black w-14 h-8 flex items-center p-1 shrink-0"
        style={{
          backgroundColor: checked ? "#04771B" : "#E8E3D8",
          justifyContent: checked ? "flex-end" : "flex-start",
        }}
        whileTap={{ scale: 0.95 }}
      >
        <motion.div
          className="w-5 h-5 border-2 border-black bg-black"
          layout
          transition={{ type: "spring", stiffness: 700, damping: 30 }}
        />
      </motion.button>
    </div>
  );
}