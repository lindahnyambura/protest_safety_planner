import { Badge } from './ui/badge';
import { MapPin, Users, Shield, Heart } from 'lucide-react';
import { ImageWithFallback } from './assets/ImageWithFallback';
import { motion } from 'motion/react';

export default function ActivityPage() {
  const pastIncidents = [
    { 
      title: 'End Femicide March', 
      status: 'peaceful', 
      date: 'Jan 27, 2024', 
      location: 'Jevanjee Gardens - Nairobi CBD',
      image: 'assets/27-01-24.jpg'
    },
    { 
      title: '#RMG - Occupy Parliament', 
      status: 'tense', 
      date: 'June 25, 2024', 
      location: 'Countrywide',
      image: 'assets/ruto-must-go.jpg'
    },
    { 
      title: 'Reject Finance Bill 2024', 
      status: 'tense', 
      date: 'June 18, 2024', 
      location: 'Nairobi CBD',
      image: 'assets/reject.jpg'
    }
  ];

  return (
    <div className="h-full overflow-y-auto pb-20" style={{ backgroundColor: '#e6e6e6' }}>
      <div className="px-6 pt-12 pb-6">
        {/* Header */}
        <motion.div 
          className="mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, ease: 'easeOut' }}
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 bg-neutral-900 rounded-2xl flex items-center justify-center">
              <MapPin className="w-6 h-6 text-white" strokeWidth={2} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">SafeNav</h1>
              <p className="text-sm text-neutral-600">Event Activity</p>
            </div>
          </div>
        </motion.div>

        {/* Happening Now Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <h3 className="mb-4 text-neutral-900 font-bold text-xl">Happening Now</h3>

          {/* Hero Card with Image */}
          <motion.div 
            className="bg-white rounded-2xl overflow-hidden border-2 border-neutral-200 mb-8 shadow-sm"
            whileHover={{ scale: 1.02 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <div className="relative h-48 overflow-hidden">
              <ImageWithFallback
                src="assets/07-07-25.jpg"
                alt="Saba Saba March"
                className="w-full h-full object-cover"
              />
              <div className="absolute top-3 right-3">
                <Badge className="bg-red-500 text-white border-0">
                  Tense
                </Badge>
              </div>
            </div>
            
            <div className="p-5">
              <h2 className="mb-3 text-neutral-900 font-semibold text-lg">
                Saba Saba March — Jamuhuri Grounds
              </h2>
              
              <div className="flex items-center gap-4 mb-3 text-neutral-600 flex-wrap">
                <span className="text-sm">July 7, 2025 • 10:00 AM</span>
                <div className="flex items-center gap-1">
                  <MapPin className="w-4 h-4 text-black" strokeWidth={2} />
                  <span className="text-sm">Jamuhuri Grounds</span>
                </div>
              </div>
              
              <p className="text-neutral-700 mb-4">
                Situation is tense with visible police presence and barricades.
                Participants advised to remain cautious near Thika Road.
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
                    <span className="text-sm text-red-600 font-medium">Caution</span>
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
          <h3 className="mb-4 text-neutral-900 font-bold text-xl">Recent Activity</h3>
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
                          : 'border-red-500 bg-red-50 text-red-700'
                      }`}
                    >
                      {incident.status === 'peaceful' ? 'Peaceful' : 'Tense'}
                    </Badge>
                  </div>
                </div>
                
                <div className="p-4">
                  <h4 className="text-neutral-900 mb-2 font-semibold">{incident.title}</h4>
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

      <style jsx>{`
        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
      `}</style>
    </div>
  );
}
