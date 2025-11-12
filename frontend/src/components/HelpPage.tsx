import { Button } from './ui/button';
import { ArrowLeft, Shield, Phone, Map, Bell, MessageCircle, CheckCircle } from 'lucide-react';
import { motion } from 'motion/react';

interface HelpPageProps {
  onBack: () => void;
}

export default function HelpPage({ onBack }: HelpPageProps) {
  const safetyChecklist = [
    'Keep your phone charged and bring a portable battery',
    'Share your plans with a trusted friend or family member',
    'Know the planned route and exit points before you go',
    'Stay aware of your surroundings at all times',
    'Follow instructions from event organizers',
    'Keep emergency contacts easily accessible',
    'Document your emergency medical information',
    'Bring water and any necessary medications',
  ];

  const emergencyContacts = [
    { name: 'Emergency Services', number: '999', type: 'General' },
    { name: 'Police', number: '112', type: 'Law Enforcement' },
    { name: 'Red Cross', number: '1199', type: 'Medical' },
    { name: 'Legal Aid Hotline', number: '1195', type: 'Legal Support' },
  ];

  const appTips = [
    {
      icon: Map,
      title: 'Check the map before you go',
      description: 'Review current conditions and identify safe zones near your intended location.',
    },
    {
      icon: Bell,
      title: 'Enable real-time alerts',
      description: 'Turn on notifications to receive immediate updates about changing conditions.',
    },
    {
      icon: Shield,
      title: 'Report what you see',
      description: 'Your anonymous reports help keep everyone safe. Report conditions honestly and accurately.',
    },
    {
      icon: MessageCircle,
      title: 'Trust the community',
      description: 'SafeNav relies on community reports. The more people contribute, the safer everyone becomes.',
    },
  ];

  return (
    <div className="h-full flex flex-col overflow-y-auto" style={{ backgroundColor: '#e6e6e6' }}>
      {/* Header */}
      <motion.div 
        className="px-6 py-4 border-b border-neutral-200 sticky top-0 z-10"
        style={{ backgroundColor: '#e6e6e6' }}
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
          <h2 className="font-bold text-xl">Help & Safety Tips</h2>
        </div>
      </motion.div>

      {/* Content */}
      <div className="flex-1 px-6 py-6 space-y-8">
        {/* Safety Checklist */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="w-5 h-5 text-neutral-600" strokeWidth={1.5} />
            <h3 className="text-neutral-900 font-semibold text-lg">Safety Checklist</h3>
          </div>
          <div className="space-y-3">
            {safetyChecklist.map((item, idx) => (
              <div key={idx} className="flex items-start gap-3 p-3 bg-neutral-50 rounded-lg border border-neutral-200">
                <div className="w-5 h-5 rounded border-2 border-neutral-300 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <div className="w-2 h-2 bg-neutral-400 rounded-sm" />
                </div>
                <p className="text-sm text-neutral-700 flex-1">{item}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Emergency Contacts */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Phone className="w-5 h-5 text-neutral-600" strokeWidth={1.5} />
            <h3 className="text-neutral-900 font-semibold text-lg">Emergency Contacts</h3>
          </div>
          <div className="space-y-2">
            {emergencyContacts.map((contact, idx) => (
              <div key={idx} className="p-4 bg-white border-2 border-neutral-200 rounded-lg">
                <div className="flex items-center justify-between mb-1">
                  <h4 className="text-neutral-900">{contact.name}</h4>
                  <a 
                    href={`tel:${contact.number}`}
                    className="px-4 py-2 bg-neutral-900 text-white rounded-lg hover:bg-neutral-800 transition-colors"
                  >
                    {contact.number}
                  </a>
                </div>
                <p className="text-sm text-neutral-600">{contact.type}</p>
              </div>
            ))}
          </div>

          <div className="mt-4 bg-amber-50 border border-amber-300 rounded-lg p-4">
            <p className="text-sm text-amber-800">
              Always call emergency services (999) if you are in immediate danger.
            </p>
          </div>
        </section>

        {/* Using This App */}
        <section>
          <div className="flex items-center gap-2 mb-4">
            <Shield className="w-5 h-5 text-neutral-600" strokeWidth={1.5} />
            <h3 className="text-neutral-900">Using This App</h3>
          </div>
          <div className="space-y-4">
            {appTips.map((tip, idx) => {
              const Icon = tip.icon;
              return (
                <div key={idx} className="flex items-start gap-4 p-4 bg-neutral-50 rounded-lg border border-neutral-200">
                  <div className="w-10 h-10 rounded-lg bg-white border border-neutral-200 flex items-center justify-center flex-shrink-0">
                    <Icon className="w-5 h-5 text-neutral-600" strokeWidth={1.5} />
                  </div>
                  <div className="flex-1">
                    <h4 className="text-neutral-900 mb-1">{tip.title}</h4>
                    <p className="text-sm text-neutral-700">{tip.description}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Additional Tips */}
        <section>
          <h3 className="mb-4 text-neutral-900">General Safety Advice</h3>
          <div className="space-y-3">
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
              <h4 className="text-green-900 mb-2">Stay Connected</h4>
              <p className="text-sm text-green-800">
                Keep in regular contact with friends. Use a buddy system and check in frequently.
              </p>
            </div>

            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <h4 className="text-blue-900 mb-2">Know Your Rights</h4>
              <p className="text-sm text-blue-800">
                Familiarize yourself with your rights to peaceful assembly and what to do if approached by authorities.
              </p>
            </div>

            <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg">
              <h4 className="text-purple-900 mb-2">Trust Your Instincts</h4>
              <p className="text-sm text-purple-800">
                If something doesn't feel right, leave the area immediately. Your safety is the top priority.
              </p>
            </div>
          </div>
        </section>

        {/* Support */}
        <section className="pb-8">
          <h3 className="mb-4 text-neutral-900">Need More Help?</h3>
          <div className="space-y-2">
            <a href="#" className="block p-4 bg-white border-2 border-neutral-200 rounded-lg hover:bg-neutral-50 transition-colors">
              <p className="text-neutral-900 mb-1">View FAQ</p>
              <p className="text-sm text-neutral-600">Common questions and answers</p>
            </a>
            <a href="#" className="block p-4 bg-white border-2 border-neutral-200 rounded-lg hover:bg-neutral-50 transition-colors">
              <p className="text-neutral-900 mb-1">Contact Support</p>
              <p className="text-sm text-neutral-600">Get in touch with our team</p>
            </a>
            <a href="#" className="block p-4 bg-white border-2 border-neutral-200 rounded-lg hover:bg-neutral-50 transition-colors">
              <p className="text-neutral-900 mb-1">Report a Bug</p>
              <p className="text-sm text-neutral-600">Help us improve SafeNav</p>
            </a>
          </div>
        </section>
      </div>
    </div>
  );
}
