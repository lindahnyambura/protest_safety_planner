import { Button } from './ui/button';
import { Navigation } from 'lucide-react';
import { motion } from 'motion/react';

interface LandingPageProps {
  onContinue: () => void;
}

export default function LandingPage({ onContinue }: LandingPageProps) {
  return (
    <div className="h-full flex flex-col justify-between" style={{ backgroundColor: '#e6e6e6' }}>
      <div className="flex-1 flex items-center justify-center px-6">
        <motion.div 
          className="text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <motion.div 
            className="w-24 h-24 mx-auto mb-6 bg-neutral-900 rounded-3xl flex items-center justify-center"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
          >
            <Navigation className="w-12 h-12 text-white" strokeWidth={2} />
          </motion.div>
          
          <motion.h1 
            className="text-5xl font-bold mb-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            SafeNav
          </motion.h1>
          
          <motion.p 
            className="text-xl text-neutral-600 mb-12"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            Community safety, together.
          </motion.p>

          <motion.div
            className="space-y-6 text-neutral-700"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}
          >
            <div className="space-y-2">
              <p className="text-lg">Navigate protests safely</p>
              <p className="text-lg">Real-time community updates</p>
              <p className="text-lg">Privacy-first design</p>
            </div>
          </motion.div>
        </motion.div>
      </div>

      {/* Fixed Bottom Button */}
      <div className="px-6 py-8 border-t border-neutral-200">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1 }}
        >
          <Button 
            onClick={onContinue}
            className="w-full bg-neutral-900 hover:bg-neutral-800 transition-all"
            size="lg"
            asChild
          >
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              Get Started
            </motion.button>
          </Button>
          
          <p className="text-xs text-neutral-500 text-center mt-4">
            Open source • No tracking • Community verified
          </p>
        </motion.div>
      </div>
    </div>
  );
}