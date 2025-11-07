import { Button } from './ui/button';
import { ArrowLeft, Lock, Eye, Trash2 } from 'lucide-react';
import { motion } from 'motion/react';

interface EthicsPageProps {
  onBack: () => void;
}

export default function EthicsPage({ onBack }: EthicsPageProps) {
  return (
    <div className="h-full flex flex-col bg-white overflow-y-auto">
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
          <h2>Ethics & DPIA</h2>
        </div>
      </motion.div>

      {/* Content */}
      <div className="flex-1 px-6 py-6 space-y-8">
        {/* Why We Built SafeNav */}
        <section>
          <h3 className="mb-4 text-neutral-900">Why We Built SafeNav</h3>
          <div className="prose prose-sm text-neutral-700 space-y-3">
            <p>
              SafeNav was created to help communities stay safe during civic events and protests. We believe that access to real-time safety information is a fundamental right, especially for those exercising their right to peaceful assembly.
            </p>
            <p>
              Our mission is to provide transparent, community-driven safety tools that respect privacy and dignity while keeping people informed and protected.
            </p>
          </div>
        </section>

        {/* How We Handle Your Data */}
        <section>
          <h3 className="mb-4 text-neutral-900">How We Handle Your Data</h3>
          <div className="space-y-4">
            <div className="bg-neutral-50 rounded-lg p-4 border border-neutral-200">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-green-100 border border-green-300 flex items-center justify-center flex-shrink-0">
                  <Lock className="w-5 h-5 text-green-700" strokeWidth={1.5} />
                </div>
                <div className="flex-1">
                  <h4 className="text-neutral-900 mb-2">No Permanent Storage</h4>
                  <p className="text-sm text-neutral-700">
                    Your location and reports are never stored permanently. All data automatically expires after 10 minutes.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-neutral-50 rounded-lg p-4 border border-neutral-200">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-100 border border-blue-300 flex items-center justify-center flex-shrink-0">
                  <Eye className="w-5 h-5 text-blue-700" strokeWidth={1.5} />
                </div>
                <div className="flex-1">
                  <h4 className="text-neutral-900 mb-2">Complete Anonymity</h4>
                  <p className="text-sm text-neutral-700">
                    All reports are submitted anonymously with no identifying information. We cannot trace reports back to individuals.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-neutral-50 rounded-lg p-4 border border-neutral-200">
              <div className="flex items-start gap-3">
                <div className="w-10 h-10 rounded-lg bg-purple-100 border border-purple-300 flex items-center justify-center flex-shrink-0">
                  <Trash2 className="w-5 h-5 text-purple-700" strokeWidth={1.5} />
                </div>
                <div className="flex-1">
                  <h4 className="text-neutral-900 mb-2">Local-First Processing</h4>
                  <p className="text-sm text-neutral-700">
                    Route calculations and safety assessments happen on your device. Minimal data is sent to servers.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Data Protection Impact Summary */}
        <section>
          <h3 className="mb-4 text-neutral-900">Data Protection Impact Summary</h3>
          <div className="bg-white border-2 border-neutral-200 rounded-xl p-6">
            <div className="space-y-6">
              {/* Collect */}
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-full bg-neutral-900 text-white flex items-center justify-center">
                  <span>1</span>
                </div>
                <div className="flex-1">
                  <h4 className="text-neutral-900 mb-2">Collect</h4>
                  <p className="text-sm text-neutral-700 mb-2">
                    Anonymous location data and safety reports from the community
                  </p>
                  <div className="text-xs text-neutral-500 space-y-1">
                    <p>• No names, emails, or phone numbers</p>
                    <p>• Location data rounded to protect privacy</p>
                    <p>• Optional metadata only</p>
                  </div>
                </div>
              </div>

              <div className="border-l-2 border-neutral-200 h-8 ml-6" />

              {/* Use */}
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-full bg-neutral-900 text-white flex items-center justify-center">
                  <span>2</span>
                </div>
                <div className="flex-1">
                  <h4 className="text-neutral-900 mb-2">Use</h4>
                  <p className="text-sm text-neutral-700 mb-2">
                    Aggregate data to provide real-time safety information and route planning
                  </p>
                  <div className="text-xs text-neutral-500 space-y-1">
                    <p>• Generate safety heat maps</p>
                    <p>• Calculate optimal routes</p>
                    <p>• Send relevant alerts</p>
                  </div>
                </div>
              </div>

              <div className="border-l-2 border-neutral-200 h-8 ml-6" />

              {/* Expire */}
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-12 h-12 rounded-full bg-neutral-900 text-white flex items-center justify-center">
                  <span>3</span>
                </div>
                <div className="flex-1">
                  <h4 className="text-neutral-900 mb-2">Expire</h4>
                  <p className="text-sm text-neutral-700 mb-2">
                    Automatic deletion ensures data is never kept longer than necessary
                  </p>
                  <div className="text-xs text-neutral-500 space-y-1">
                    <p>• All data expires after 10 minutes</p>
                    <p>• No historical database</p>
                    <p>• Complete data removal</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Important Notice */}
        <section>
          <div className="bg-amber-50 border border-amber-300 rounded-lg p-4">
            <h4 className="text-amber-900 mb-2">Important Notice</h4>
            <p className="text-sm text-amber-800">
              SafeNav is designed for community safety coordination and should not be used to collect personally identifiable information (PII) or secure sensitive data. This is a transparency tool, not a secure communications platform.
            </p>
          </div>
        </section>

        {/* Additional Resources */}
        <section className="pb-8">
          <h3 className="mb-4 text-neutral-900">Learn More</h3>
          <div className="space-y-2">
            <a href="#" className="block p-3 bg-neutral-50 rounded-lg border border-neutral-200 hover:bg-neutral-100 transition-colors">
              <p className="text-sm text-neutral-900">View Full Privacy Policy</p>
            </a>
            <a href="#" className="block p-3 bg-neutral-50 rounded-lg border border-neutral-200 hover:bg-neutral-100 transition-colors">
              <p className="text-sm text-neutral-900">Read Data Protection Impact Assessment</p>
            </a>
            <a href="#" className="block p-3 bg-neutral-50 rounded-lg border border-neutral-200 hover:bg-neutral-100 transition-colors">
              <p className="text-sm text-neutral-900">Contact Privacy Team</p>
            </a>
          </div>
        </section>
      </div>
    </div>
  );
}
