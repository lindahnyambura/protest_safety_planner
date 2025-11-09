import { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { ArrowLeft, Wind, Shield, Users, CheckCircle, Droplets, MapPin, Clock } from 'lucide-react';
import { motion } from 'motion/react';

interface AlertsFeedProps {
  onBack: () => void;
  onAlertClick?: (lat: number, lng: number) => void; // Navigate to location on map
}

interface Report {
  id: string;
  type: 'safe' | 'crowd' | 'police' | 'tear_gas' | 'water_cannon';
  lat: number;
  lng: number;
  confidence: number;
  timestamp: number;
  expires_at: number;
  node_id: string;
  location_name?: string;
}

export default function AlertsFeed({ onBack, onAlertClick }: AlertsFeedProps) {
  console.log('[AlertsFeed] Component rendered');
  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const [reports, setReports] = useState<Report[]>([]);
  const [loading, setLoading] = useState(true);
  const API_BASE_URL = import.meta.env.VITE_API_URL;

  useEffect(() => {
    fetchReports();
    
    // Refresh every 15 seconds
    const interval = setInterval(fetchReports, 15000);
    
    return () => clearInterval(interval);
  }, []);

  const fetchReports = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/reports/active`);
      
      if (response.ok) {
        const data = await response.json();
        console.log('[AlertsFeed] Fetched reports:', data.reports?.length || 0);
        setReports(data.reports || []);
      }
    } catch (error) {
      console.error('[AlertsFeed] Failed to fetch reports:', error);
    } finally {
      setLoading(false);
    }
  };

  const filters = [
    { id: 'tear_gas', label: 'Tear Gas', icon: Wind },
    { id: 'water_cannon', label: 'Water Cannon', icon: Droplets },
    { id: 'police', label: 'Police', icon: Shield },
    { id: 'crowd', label: 'Crowd', icon: Users },
    { id: 'safe', label: 'Safe', icon: CheckCircle },
  ];

  const filteredReports = activeFilter
    ? reports.filter(report => report.type === activeFilter)
    : reports;

  const getReportColor = (type: string) => {
    switch (type) {
      case 'tear_gas':
      case 'water_cannon':
        return 'red';
      case 'police':
        return 'blue';
      case 'crowd':
        return 'amber';
      case 'safe':
        return 'green';
      default:
        return 'neutral';
    }
  };

  const getReportIcon = (type: string) => {
    switch (type) {
      case 'tear_gas':
        return Wind;
      case 'water_cannon':
        return Droplets;
      case 'police':
        return Shield;
      case 'crowd':
        return Users;
      case 'safe':
        return CheckCircle;
      default:
        return MapPin;
    }
  };

  const getReportTitle = (type: string) => {
    return type.replace('_', ' ').split(' ').map(w => 
      w.charAt(0).toUpperCase() + w.slice(1)
    ).join(' ');
  };

  const getTimeAgo = (timestamp: number) => {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    
    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
  };

  const getExpiresIn = (expiresAt: number) => {
    const seconds = Math.floor((expiresAt - Date.now()) / 1000);
    
    if (seconds <= 0) return 'Expired';
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    return `${Math.floor(seconds / 3600)}h`;
  };

  const handleReportClick = (report: Report) => {
    console.log('[AlertsFeed] Report clicked:', report);
    
    if (onAlertClick) {
      onAlertClick(report.lat, report.lng);
    }
    
    onBack(); // Return to map after clicking
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
          <div className="flex-1">
            <h2 className="text-xl font-bold">Alerts & Events</h2>
            <p className="text-sm text-neutral-600">
              {loading ? 'Loading...' : `${reports.length} active report${reports.length !== 1 ? 's' : ''}`}
            </p>
          </div>
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
              All ({reports.length})
            </Badge>
          </motion.div>
          {filters.map((filter) => {
            const Icon = filter.icon;
            const count = reports.filter(r => r.type === filter.id).length;
            
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
                  {filter.label} ({count})
                </Badge>
              </motion.div>
            );
          })}
        </div>
      </motion.div>

      {/* Alerts List */}
      <div className="flex-1 overflow-y-auto">
        {loading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="w-12 h-12 border-4 border-neutral-200 border-t-neutral-900 rounded-full animate-spin mx-auto mb-3"></div>
              <p className="text-neutral-500">Loading reports...</p>
            </div>
          </div>
        ) : filteredReports.length === 0 ? (
          <div className="flex items-center justify-center h-full px-6 text-center">
            <div>
              <MapPin className="w-12 h-12 text-neutral-300 mx-auto mb-3" strokeWidth={1.5} />
              <p className="text-neutral-500">
                {activeFilter ? 'No reports in this category' : 'No active reports'}
              </p>
              <p className="text-sm text-neutral-400 mt-2">
                Reports will appear here as they are submitted
              </p>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-neutral-200">
            {filteredReports.map((report, idx) => {
              const Icon = getReportIcon(report.type);
              const color = getReportColor(report.type);
              const timeAgo = getTimeAgo(report.timestamp);
              const expiresIn = getExpiresIn(report.expires_at);
              const isExpiring = (report.expires_at - Date.now()) < 120000; // Less than 2 minutes
              
              return (
                <motion.button
                  key={report.id}
                  onClick={() => handleReportClick(report)}
                  className="w-full px-6 py-4 hover:bg-neutral-50 transition-colors text-left"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  whileTap={{ scale: 0.99 }}
                >
                  <div className="flex items-start gap-3">
                    <div className={`w-10 h-10 rounded-lg border-2 flex items-center justify-center flex-shrink-0 ${
                      color === 'red' ? 'border-red-400 bg-red-50' :
                      color === 'blue' ? 'border-blue-400 bg-blue-50' :
                      color === 'amber' ? 'border-amber-400 bg-amber-50' :
                      color === 'green' ? 'border-green-400 bg-green-50' :
                      'border-neutral-400 bg-neutral-50'
                    }`}>
                      <Icon className={`w-5 h-5 ${
                        color === 'red' ? 'text-red-600' :
                        color === 'blue' ? 'text-blue-600' :
                        color === 'amber' ? 'text-amber-600' :
                        color === 'green' ? 'text-green-600' :
                        'text-neutral-600'
                      }`} strokeWidth={1.5} />
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-2 mb-1">
                        <h4 className="text-neutral-900 font-medium">
                          {getReportTitle(report.type)}
                        </h4>
                        <Badge
                          variant="outline"
                          className={`flex-shrink-0 text-xs ${
                            report.confidence > 0.7
                              ? 'border-green-300 text-green-700'
                              : report.confidence > 0.4
                              ? 'border-amber-300 text-amber-700'
                              : 'border-neutral-300 text-neutral-600'
                          }`}
                        >
                          {Math.round(report.confidence * 100)}%
                        </Badge>
                      </div>
                      
                      <div className="flex items-center gap-2 text-sm text-neutral-600 mb-2">
                        <MapPin className="w-3 h-3" strokeWidth={1.5} />
                        <span className="truncate">
                          {report.location_name || `${report.lat.toFixed(4)}, ${report.lng.toFixed(4)}`}
                        </span>
                      </div>

                      <div className="flex items-center gap-3 text-xs text-neutral-500">
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {timeAgo}
                        </span>
                        <span className={`${isExpiring ? 'text-amber-600 font-medium' : ''}`}>
                          Expires in {expiresIn}
                        </span>
                      </div>
                    </div>
                  </div>
                </motion.button>
              );
            })}
          </div>
        )}
      </div>

      {/* Info Footer */}
      {!loading && filteredReports.length > 0 && (
        <div className="px-6 py-4 bg-neutral-50 border-t border-neutral-200">
          <p className="text-sm text-neutral-600 text-center">
            Tap any alert to view its location on the map
          </p>
        </div>
      )}
    </div>
  );
}