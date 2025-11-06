import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { MapPin, Users, Shield, Heart, Navigation, TrendingUp } from 'lucide-react';
import { ImageWithFallback } from './figma/ImageWithFallback';
import { motion } from 'motion/react';

interface LandingPageProps {
  onContinue: () => void;
}

export default function LandingPage({ onContinue }: LandingPageProps) {
  const pastIncidents = [
    { 
      title: 'Healthcare Workers Rally', 
      status: 'peaceful', 
      date: 'Nov 3, 2025', 
      location: 'Uhuru Park',
      image: 'https://images.unsplash.com/photo-1610629315052-48f40742d3ae?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxoZWFsdGhjYXJlJTIwd29ya2VycyUyMHJhbGx5fGVufDF8fHx8MTc2MjQwMjcwNnww&ixlib=rb-4.1.0&q=80&w=1080'
    },
    { 
      title: 'Student Demonstration', 
      status: 'tense', 
      date: 'Oct 28, 2025', 
      location: 'University Way',
      image: 'https://images.unsplash.com/photo-1759061729194-e1aa695a4861?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxzdHVkZW50JTIwZGVtb25zdHJhdGlvbiUyMHJhbGx5fGVufDF8fHx8MTc2MjQwMjcwNnww&ixlib=rb-4.1.0&q=80&w=1080'
    },
    { 
      title: 'Climate Action March', 
      status: 'peaceful', 
      date: 'Oct 20, 2025', 
      location: 'CBD',
      image: 'https://images.unsplash.com/photo-1759078528289-86c242c3734b?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjbGltYXRlJTIwYWN0aW9uJTIwbWFyY2h8ZW58MXx8fHwxNzYyNDAyNzA3fDA&ixlib=rb-4.1.0&q=80&w=1080'
    },
  ];

  return (
    <div className="h-full overflow-y-auto bg-white">
      <div className="px-6 pt-12 pb-32">
        {/* Logo and App Name */}
        <motion.div 
          className="text-center mb-12"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        >
          <div className="w-20 h-20 mx-auto mb-4 bg-neutral-900 rounded-3xl flex items-center justify-center">
            <Navigation className="w-10 h-10 text-white" strokeWidth={2} />
          </div>
          
          <h1 className="text-4xl font-bold mb-2">SafeNav</h1>
          <p className="text-neutral-600">
            Community safety, together.
          </p>
        </motion.div>

        {/* Happening Now Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <h3 className="mb-4 text-neutral-900">Happening Now</h3>

          {/* Hero Card with Image */}
          <motion.div 
            className="bg-white rounded-2xl overflow-hidden border-2 border-neutral-200 mb-8 shadow-sm"
            whileHover={{ scale: 1.02 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <div className="relative h-48 overflow-hidden">
              <ImageWithFallback
                src="https://images.unsplash.com/photo-1759117708874-029ce6e1eb65?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwcm90ZXN0JTIwbWFyY2glMjBjcm93ZCUyMHBlYWNlZnVsfGVufDF8fHx8MTc2MjQwMjcwNXww&ixlib=rb-4.1.0&q=80&w=1080"
                alt="March for Justice"
                className="w-full h-full object-cover"
              />
              <div className="absolute top-3 right-3">
                <Badge className="bg-green-500 text-white border-0">
                  Peaceful
                </Badge>
              </div>
            </div>
            
            <div className="p-5">
              <h2 className="mb-3 text-neutral-900">March for Justice — Nairobi CBD</h2>
              
              <div className="flex items-center gap-4 mb-3 text-neutral-600 flex-wrap">
                <span className="text-sm">Nov 6, 2025 • 2:00 PM</span>
                <div className="flex items-center gap-1">
                  <MapPin className="w-4 h-4 text-black" strokeWidth={2} />
                  <span className="text-sm">Kenyatta Ave – CBD</span>
                </div>
              </div>
              
              <p className="text-neutral-700 mb-4">
                Peaceful gathering progressing along planned route. Community observers report calm atmosphere with organized coordination.
              </p>
              
              <div className="flex items-center justify-between pt-4 border-t border-neutral-200">
                <span className="text-sm text-neutral-500">At a Glance</span>
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-1">
                    <Users className="w-4 h-4 text-black" strokeWidth={2} />
                    <span className="text-sm text-neutral-600">2.5k</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Shield className="w-4 h-4 text-black" strokeWidth={2} />
                    <span className="text-sm text-neutral-600">Safe</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Heart className="w-4 h-4 text-black" strokeWidth={2} />
                    <span className="text-sm text-neutral-600">Medics</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>

        {/* Recent Activity */}
        <motion.div 
          className="mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <h3 className="mb-4 text-neutral-900">Recent Activity</h3>
          <div className="flex gap-4 overflow-x-auto scrollbar-hide pb-2" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
            {pastIncidents.map((incident, idx) => (
              <motion.div
                key={idx}
                className="flex-shrink-0 w-64 bg-white border-2 border-neutral-200 rounded-2xl overflow-hidden shadow-sm"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ 
                  duration: 0.5, 
                  delay: 0.5 + idx * 0.1,
                  type: 'spring',
                  stiffness: 100
                }}
                whileHover={{ y: -4 }}
              >
                <div className="relative h-32 overflow-hidden">
                  <ImageWithFallback
                    src={incident.image}
                    alt={incident.title}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-2 right-2">
                    <Badge 
                      variant="outline"
                      className={`border-2 ${
                        incident.status === 'peaceful'
                          ? 'border-green-500 bg-green-50 text-green-700'
                          : 'border-amber-500 bg-amber-50 text-amber-700'
                      }`}
                    >
                      {incident.status === 'peaceful' ? 'Peaceful' : 'Tense'}
                    </Badge>
                  </div>
                </div>
                
                <div className="p-4">
                  <h4 className="text-neutral-900 mb-2">{incident.title}</h4>
                  <p className="text-sm text-neutral-500 mb-1">{incident.date}</p>
                  <p className="text-sm text-neutral-600 flex items-center gap-1">
                    <MapPin className="w-3 h-3 text-black" strokeWidth={2} />
                    {incident.location}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Fixed Bottom Section */}
      <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-neutral-200 px-6 py-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
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
              Continue to Map
            </motion.button>
          </Button>
          
          <p className="text-xs text-neutral-500 text-center mt-4">
            Verified by community observers and partners.
          </p>
        </motion.div>
      </div>

      <style jsx>{`
        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
      `}</style>
    </div>
  );
}
