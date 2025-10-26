import { motion } from "motion/react";
import { ArrowLeft, Shield, Database, Lock, Clock, FileText, Mail } from "lucide-react";

interface EthicsProps {
  onBack: () => void;
}

export function Ethics({ onBack }: EthicsProps) {
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
        <h1 className="text-lg">ETHICS & DPIA</h1>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Hero Section */}
        <div className="border-3 border-black bg-[#04771B] p-6 text-center">
          <Shield size={40} className="mx-auto mb-3" />
          <h2 className="text-xl mb-2" style={{ fontWeight: 700 }}>Data Protection Impact Assessment</h2>
          <p className="text-sm">Transparent. Ethical. Privacy-First.</p>
        </div>

        {/* Purpose */}
        <section className="border-2 border-black bg-[#FDF8F0] p-4">
          <div className="flex items-center gap-2 mb-3">
            <FileText size={20} />
            <h3 className="text-sm" style={{ fontWeight: 700 }}>PURPOSE</h3>
          </div>
          <p className="text-sm leading-relaxed">
            SafeNav exists to protect protesters exercising their democratic rights. 
            We compute safe routes during demonstrations using crowd-sourced hazard reports. 
            The system prioritizes user privacy and operates without centralized surveillance.
          </p>
        </section>

        {/* What Data We Process */}
        <section className="border-2 border-black bg-[#E8E3D8] p-4">
          <div className="flex items-center gap-2 mb-3">
            <Database size={20} />
            <h3 className="text-sm" style={{ fontWeight: 700 }}>WHAT DATA WE PROCESS</h3>
          </div>
          <ul className="space-y-2 text-sm">
            <li className="flex gap-2">
              <span className="shrink-0">•</span>
              <span><strong>Location data:</strong> Your GPS coordinates for routing and hazard proximity (processed locally)</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">•</span>
              <span><strong>Hazard reports:</strong> Type, location, confidence level, optional text description</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">•</span>
              <span><strong>Timestamps:</strong> When hazards were reported (for TTL enforcement)</span>
            </li>
          </ul>
        </section>

        {/* What We Do NOT Collect */}
        <section className="border-2 border-black bg-[#FDF8F0] p-4">
          <div className="flex items-center gap-2 mb-3">
            <Lock size={20} />
            <h3 className="text-sm" style={{ fontWeight: 700 }}>WHAT WE DO NOT COLLECT</h3>
          </div>
          <ul className="space-y-2 text-sm">
            <li className="flex gap-2">
              <span className="shrink-0">✗</span>
              <span>User accounts or persistent identifiers</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">✗</span>
              <span>Device IDs, IP addresses, or fingerprints</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">✗</span>
              <span>Movement history or route logs</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">✗</span>
              <span>Personal identifiable information (PII)</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">✗</span>
              <span>Third-party analytics or tracking scripts</span>
            </li>
          </ul>
        </section>

        {/* Data Lifecycle */}
        <section className="border-3 border-black bg-[#19647E] p-4">
          <div className="flex items-center gap-2 mb-3">
            <Clock size={20} />
            <h3 className="text-sm" style={{ fontWeight: 700 }}>DATA LIFECYCLE</h3>
          </div>
          <div className="space-y-3 text-sm">
            <div className="border-2 border-black bg-[#FDF8F0] p-3">
              <p className="mb-2" style={{ fontWeight: 600 }}>Timeline:</p>
              <ol className="space-y-1 text-xs">
                <li><strong>1.</strong> Report created → anonymized immediately</li>
                <li><strong>2.</strong> Shared with nearby users (no origin metadata)</li>
                <li><strong>3.</strong> Used in live hazard simulation</li>
                <li><strong>4.</strong> Auto-deleted after TTL (2, 5, or 10 minutes)</li>
              </ol>
            </div>
            <p className="text-xs opacity-80">
              <strong>Note:</strong> You control TTL in Settings. Shorter TTL = less data persistence, 
              but may reduce community awareness of evolving situations.
            </p>
          </div>
        </section>

        {/* Legal & Ethical Basis */}
        <section className="border-2 border-black bg-[#FDF8F0] p-4">
          <div className="flex items-center gap-2 mb-3">
            <Shield size={20} />
            <h3 className="text-sm" style={{ fontWeight: 700 }}>LEGAL & ETHICAL BASIS</h3>
          </div>
          <div className="space-y-2 text-sm">
            <p>
              <strong>Legitimate Interest:</strong> Public safety during protests (balanced against privacy rights)
            </p>
            <p>
              <strong>Data Minimization:</strong> We collect only what's essential for routing
            </p>
            <p>
              <strong>Consent:</strong> Optional reporting — users choose whether to contribute data
            </p>
            <p>
              <strong>Security:</strong> End-to-end processing on device, no cloud storage of personal data
            </p>
          </div>
        </section>

        {/* Your Controls */}
        <section className="border-2 border-black bg-[#E8E3D8] p-4">
          <div className="flex items-center gap-2 mb-3">
            <Lock size={20} />
            <h3 className="text-sm" style={{ fontWeight: 700 }}>YOUR CONTROLS</h3>
          </div>
          <ul className="space-y-2 text-sm">
            <li className="flex gap-2">
              <span className="shrink-0">→</span>
              <span>Toggle report sharing on/off in Settings</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">→</span>
              <span>Set custom TTL (2/5/10 minutes)</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">→</span>
              <span>Delete all local data with one tap</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">→</span>
              <span>Use manual location mode (no GPS)</span>
            </li>
          </ul>
        </section>

        {/* Contact */}
        <section className="border-2 border-black bg-[#FDF8F0] p-4">
          <div className="flex items-center gap-2 mb-3">
            <Mail size={20} />
            <h3 className="text-sm" style={{ fontWeight: 700 }}>CONTACT & TRANSPARENCY</h3>
          </div>
          <p className="text-sm mb-2">
            Questions about data practices? Concerns about ethical use?
          </p>
          <a 
            href="mailto:privacy@safenav.app"
            className="inline-block border-2 border-black bg-black text-[#FDF8F0] px-4 py-2 text-sm hover:bg-[#04771B] hover:text-black transition-colors"
          >
            privacy@safenav.app
          </a>
          <p className="text-xs mt-3 opacity-60">
            This DPIA is version-controlled and publicly auditable. Last updated: October 2025.
          </p>
        </section>

        {/* Footer Note */}
        <div className="border-3 border-black bg-[#AE1E2A] p-4 text-center">
          <p className="text-sm" style={{ fontWeight: 600 }}>
            SafeNav is NOT intended for collecting PII or securing sensitive communications. 
            Use encrypted messaging for organizing.
          </p>
        </div>
      </div>
    </div>
  );
}
