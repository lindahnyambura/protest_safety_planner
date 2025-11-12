import { motion } from 'motion/react';
import { TrendingUp, Users, MapPin, Shield, AlertTriangle, CheckCircle, Activity } from 'lucide-react';
import { Card } from './ui/card';

export default function HomePage() {
  const stats = [
    { 
      label: 'Active Events',
      value: '3',
      change: '+1 today',
      icon: Activity,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50'
    },
    { 
      label: 'Total Attendees',
      value: '5.2k',
      change: '+800 today',
      icon: Users,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50'
    },
    { 
      label: 'Peaceful',
      value: '2',
      change: '67% of events',
      icon: CheckCircle,
      color: 'text-green-600',
      bgColor: 'bg-green-50'
    },
    { 
      label: 'Tense',
      value: '1',
      change: '33% of events',
      icon: AlertTriangle,
      color: 'text-red-600',
      bgColor: 'bg-red-50'
    },
  ];

  const topLocations = [
    { name: 'Nairobi CBD', events: 2, attendees: '3.5k', status: 'tense' },
    { name: 'Uhuru Park', events: 1, attendees: '1.2k', status: 'peaceful' },
    { name: 'University Way', events: 1, attendees: '500', status: 'peaceful' },
  ];

  const recentUpdates = [
    { 
      time: '5 min ago',
      event: 'March for Justice',
      update: 'Crowd size increased to 2.5k attendees',
      status: 'tense'
    },
    { 
      time: '22 min ago',
      event: 'Healthcare Workers Rally',
      update: 'Event concluded peacefully',
      status: 'peaceful'
    },
    { 
      time: '1 hr ago',
      event: 'Student Demonstration',
      update: 'Route diverted to University Way',
      status: 'peaceful'
    },
  ];

  return (
    <div className="h-full overflow-y-auto pb-20" style={{ backgroundColor: '#e6e6e6' }}>
      <div className="px-6 pt-12 pb-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="mb-8"
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 bg-neutral-900 rounded-2xl flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-white" strokeWidth={2} />
            </div>
            <div>
              <h1 className="text-3xl font-bold">SafeNav</h1>
              <p className="text-sm text-neutral-600">Live Protest Dashboard</p>
            </div>
          </div>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="mb-8"
        >
          <h2 className="mb-4 text-neutral-900 font-bold text-xl">Today's Overview</h2>
          <div className="grid grid-cols-2 gap-4">
            {stats.map((stat, idx) => {
              const Icon = stat.icon;
              return (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, delay: 0.2 + idx * 0.1 }}
                  whileHover={{ scale: 1.05 }}
                  className="bg-white border-2 border-neutral-200 rounded-2xl p-4 shadow-sm"
                >
                  <div className={`w-10 h-10 ${stat.bgColor} rounded-xl flex items-center justify-center mb-3`}>
                    <Icon className={`w-5 h-5 ${stat.color}`} strokeWidth={2} />
                  </div>
                  <div className="text-3xl font-bold mb-1">{stat.value}</div>
                  <div className="text-sm text-neutral-600 mb-1">{stat.label}</div>
                  <div className="text-xs text-neutral-500">{stat.change}</div>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* Top Locations */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mb-8"
        >
          <h2 className="mb-4 text-neutral-900 font-bold text-xl">Top Locations</h2>
          <div className="space-y-3">
            {topLocations.map((location, idx) => (
              <motion.div
                key={location.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.4 + idx * 0.1 }}
                whileHover={{ x: 4 }}
                className="bg-white border-2 border-neutral-200 rounded-2xl p-4 shadow-sm"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <MapPin className="w-5 h-5 text-neutral-900" strokeWidth={2} />
                    <span className="font-semibold text-neutral-900">{location.name}</span>
                  </div>
                  <span
                    className={`text-xs px-2 py-1 rounded-lg border-2 ${
                      location.status === 'peaceful'
                        ? 'border-green-500 bg-green-50 text-green-700'
                        : 'border-red-500 bg-red-50 text-red-700'
                    }`}
                  >
                    {location.status === 'peaceful' ? 'Peaceful' : 'Tense'}
                  </span>
                </div>
                <div className="flex items-center gap-4 text-sm text-neutral-600">
                  <span>{location.events} event{location.events > 1 ? 's' : ''}</span>
                  <span>•</span>
                  <span>{location.attendees} attendees</span>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Recent Updates */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="mb-4"
        >
          <h2 className="mb-4 text-neutral-900 font-bold text-xl">Recent Updates</h2>
          <div className="space-y-3">
            {recentUpdates.map((update, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.4, delay: 0.6 + idx * 0.1 }}
                className="bg-white border-2 border-neutral-200 rounded-2xl p-4 shadow-sm"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <Activity className="w-4 h-4 text-neutral-900" strokeWidth={2} />
                      <span className="font-semibold text-neutral-900 text-sm">{update.event}</span>
                    </div>
                    <p className="text-sm text-neutral-700">{update.update}</p>
                  </div>
                  <span className="text-xs text-neutral-500 ml-2">{update.time}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
