import { useState } from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ArrowLeft, Wind, Shield, Users, AlertTriangle, MapPin } from 'lucide-react';
import { motion } from 'motion/react';

interface AlertsFeedProps {
  onBack: () => void;
}

export default function AlertsFeed({ onBack }: AlertsFeedProps) {
  console.log('[AlertsFeed] Component rendered');
  const [activeFilter, setActiveFilter] = useState<string | null>(null);

  const alerts = [
    {
      id: 1,
      type: 'tear-gas',
      title: 'Tear gas deployed',
      location: 'Kenyatta Ave, CBD',
      time: '2 min ago',
      confidence: 'high',
      description: 'Multiple reports of tear gas near Kenyatta Ave intersection.',
      coordinates: { x: 45, y: 35 },
    },
    {
      id: 2,
      type: 'police',
      title: 'Police presence',
      location: 'Moi Ave',
      time: '5 min ago',
      confidence: 'medium',
      description: 'Police units observed setting up blockade.',
      coordinates: { x: 60, y: 50 },
    },
    {
      id: 3,
      type: 'crowd',
      title: 'Large crowd gathering',
      location: 'Uhuru Park entrance',
      time: '8 min ago',
      confidence: 'high',
      description: 'Peaceful gathering estimated at 1,500+ people.',
      coordinates: { x: 30, y: 60 },
    },
    {
      id: 4,
      type: 'crowd',
      title: 'Crowd dispersing',
      location: 'Tom Mboya St',
      time: '12 min ago',
      confidence: 'medium',
      description: 'Crowd moving peacefully toward designated area.',
      coordinates: { x: 55, y: 40 },
    },
    {
      id: 5,
      type: 'tear-gas',
      title: 'Tear gas reported',
      location: 'Haile Selassie Ave',
      time: '15 min ago',
      confidence: 'low',
      description: 'Unconfirmed report, awaiting verification.',
      coordinates: { x: 70, y: 30 },
    },
  ];

  const filters = [
    { id: 'tear-gas', label: 'Tear Gas', icon: Wind },
    { id: 'police', label: 'Police', icon: Shield },
    { id: 'crowd', label: 'Crowd', icon: Users },
  ];

  const filteredAlerts = activeFilter
    ? alerts.filter(alert => alert.type === activeFilter)
    : alerts;

  const getAlertColor = (type: string) => {
    switch (type) {
      case 'tear-gas':
        return 'red';
      case 'police':
        return 'blue';
      case 'crowd':
        return 'amber';
      default:
        return 'neutral';
    }
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'tear-gas':
        return Wind;
      case 'police':
        return Shield;
      case 'crowd':
        return Users;
      default:
        return AlertTriangle;
    }
  };

  return (
    <div className="h-full flex flex-col bg-white">
      {/* Header */}
      <motion.div 
        className="px-6 py-4 border-b border-neutral-200"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex items-center gap-3 mb-4">
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
          <h2>Alerts & Events</h2>
        </div>

        {/* Filter Chips */}
        <div className="flex gap-2 overflow-x-auto pb-1">
          <motion.div whileTap={{ scale: 0.95 }}>
            <Badge
              onClick={() => setActiveFilter(null)}
              className={`cursor-pointer flex-shrink-0 border-2 ${
                activeFilter === null
                  ? 'bg-neutral-900 text-white border-neutral-900'
                  : 'bg-white text-neutral-600 border-neutral-300'
              }`}
            >
              All
            </Badge>
          </motion.div>
          {filters.map((filter) => {
            const Icon = filter.icon;
            return (
              <motion.div key={filter.id} whileTap={{ scale: 0.95 }}>
                <Badge
                  onClick={() => setActiveFilter(filter.id)}
                  className={`cursor-pointer flex-shrink-0 flex items-center gap-1 border-2 ${
                    activeFilter === filter.id
                      ? 'bg-neutral-900 text-white border-neutral-900'
                      : 'bg-white text-neutral-600 border-neutral-300'
                  }`}
                >
                  <Icon className="w-3 h-3" strokeWidth={2} />
                  {filter.label}
                </Badge>
              </motion.div>
            );
          })}
        </div>
      </motion.div>

      {/* Alerts List */}
      <div className="flex-1 overflow-y-auto">
        {filteredAlerts.length === 0 ? (
          <div className="flex items-center justify-center h-full px-6 text-center">
            <div>
              <AlertTriangle className="w-12 h-12 text-neutral-300 mx-auto mb-3" strokeWidth={1.5} />
              <p className="text-neutral-500">No alerts in this category</p>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-neutral-200">
            {filteredAlerts.map((alert, idx) => {
              const Icon = getAlertIcon(alert.type);
              const color = getAlertColor(alert.type);
              
              return (
                <motion.div
                  key={alert.id}
                  className="px-6 py-4 hover:bg-neutral-50 transition-colors cursor-pointer"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  whileTap={{ scale: 0.99 }}
                >
                  <div className="flex items-start gap-3">
                    <div className={`w-10 h-10 rounded-lg border-2 flex items-center justify-center flex-shrink-0 border-${color}-400 bg-${color}-50`}>
                      <Icon className={`w-5 h-5 text-${color}-600`} strokeWidth={1.5} />
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <h4 className="text-neutral-900">{alert.title}</h4>
                        <Badge
                          variant="outline"
                          className={`flex-shrink-0 text-xs ${
                            alert.confidence === 'high'
                              ? 'border-green-300 text-green-700'
                              : alert.confidence === 'medium'
                              ? 'border-amber-300 text-amber-700'
                              : 'border-neutral-300 text-neutral-600'
                          }`}
                        >
                          {alert.confidence}
                        </Badge>
                      </div>
                      
                      <div className="flex items-center gap-2 text-sm text-neutral-600 mb-2">
                        <MapPin className="w-3 h-3" strokeWidth={1.5} />
                        <span>{alert.location}</span>
                        <span>•</span>
                        <span>{alert.time}</span>
                      </div>
                      
                      <p className="text-sm text-neutral-700">{alert.description}</p>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>

      {/* Info Footer */}
      <div className="px-6 py-4 bg-neutral-50 border-t border-neutral-200">
        <p className="text-sm text-neutral-600 text-center">
          Tap any alert to highlight location on map
        </p>
      </div>
    </div>
  );
}
