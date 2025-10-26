import { motion } from "motion/react";
import { ArrowLeft, AlertTriangle, Phone, Shield, Map, Users, Wind } from "lucide-react";

interface HelpProps {
  onBack: () => void;
}

export function Help({ onBack }: HelpProps) {
  const safetyTips = [
    {
      title: "Before Protests",
      tips: [
        "Share your plans with trusted contacts",
        "Charge your phone fully",
        "Know multiple exit routes",
        "Bring water and basic first aid",
      ],
    },
    {
      title: "During Protests",
      tips: [
        "Stay aware of your surroundings",
        "Keep moving if threatened",
        "Avoid bottlenecks and dead ends",
        "Follow the app's real-time alerts",
      ],
    },
    {
      title: "Using SafeNav",
      tips: [
        "Report hazards immediately for others",
        "Use 'Safe Route' before moving",
        "Enable auto re-route for dynamic updates",
        "All data stays on your device",
      ],
    },
  ];

  const emergencyContacts = [
    { name: "Emergency Services", number: "999" },
    { name: "Police Hotline", number: "112" },
    { name: "Red Cross", number: "1199" },
  ];

  const appFeatures = [
    {
      icon: Map,
      title: "Real-time Hazard Map",
      description: "See crowd-sourced reports of tear gas, police, and crowds",
    },
    {
      icon: Shield,
      title: "Safe Route Planning",
      description: "Compute routes that avoid reported hazards",
    },
    {
      icon: AlertTriangle,
      title: "Live Alerts",
      description: "Get notified when new dangers appear on your route",
    },
    {
      icon: Users,
      title: "Anonymous Reporting",
      description: "Help others while staying completely anonymous",
    },
  ];

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
        <h1 className="text-lg">HELP & SAFETY TIPS</h1>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Emergency Contacts */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Phone size={20} />
            <h2 className="text-base">Emergency Contacts</h2>
          </div>
          <div className="border-2 border-black bg-[#AE1E2A] p-4 space-y-3">
            {emergencyContacts.map((contact, index) => (
              <div
                key={index}
                className="flex items-center justify-between border-b-2 border-black last:border-0 pb-3 last:pb-0"
              >
                <span className="text-sm">{contact.name}</span>
                <a
                  href={`tel:${contact.number}`}
                  className="border-2 border-black bg-black text-[#FDF8F0] px-4 py-2 text-sm hover:bg-[#04771B] hover:text-black transition-colors"
                >
                  {contact.number}
                </a>
              </div>
            ))}
          </div>
        </section>

        {/* Safety Checklist */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Shield size={20} />
            <h2 className="text-base">Safety Checklist</h2>
          </div>
          <div className="space-y-4">
            {safetyTips.map((section, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="border-2 border-black bg-[#E8E3D8] p-4"
              >
                <h3 className="text-sm mb-3">{section.title}</h3>
                <ul className="space-y-2">
                  {section.tips.map((tip, tipIndex) => (
                    <li key={tipIndex} className="flex gap-2 text-sm">
                      <span className="shrink-0">•</span>
                      <span>{tip}</span>
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </section>

        {/* App Features */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Map size={20} />
            <h2 className="text-base">Using This App</h2>
          </div>
          <div className="space-y-3">
            {appFeatures.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="border-2 border-black bg-[#FDF8F0] p-4 flex gap-3"
                >
                  <div className="border-2 border-black p-2 bg-[#04771B] shrink-0">
                    <Icon size={20} />
                  </div>
                  <div>
                    <h3 className="text-sm mb-1">{feature.title}</h3>
                    <p className="text-xs opacity-70">{feature.description}</p>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </section>

        {/* Privacy Notice */}
        <section className="border-3 border-black bg-[#04771B] p-4">
          <h3 className="text-sm mb-2">Privacy & Trust</h3>
          <p className="text-xs leading-relaxed">
            SafeNav is designed with privacy as the foundation. All location processing 
            happens on your device. No accounts, no servers storing your movements, no third-party 
            tracking. Reports are anonymized before contributing to the community map.
          </p>
        </section>

        {/* Warning */}
        <section className="border-2 border-black bg-[#FDF8F0] p-4">
          <div className="flex gap-3">
            <AlertTriangle size={20} className="shrink-0" />
            <div>
              <h3 className="text-sm mb-2">Important Notice</h3>
              <p className="text-xs leading-relaxed">
                This app provides guidance based on crowd-sourced data. Always trust your own 
                judgment and prioritize your safety. In case of immediate danger, contact 
                emergency services.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}