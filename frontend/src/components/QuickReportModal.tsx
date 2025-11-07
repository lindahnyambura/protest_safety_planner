import { useState, useEffect } from 'react';
import { Button } from './ui/button';
import { X, CheckCircle, Users, Shield, Wind, Droplets } from 'lucide-react';
import { toast } from 'sonner';
import { motion } from 'motion/react';
import { apiService } from '../services/api';

interface QuickReportModalProps {
  onClose: () => void;
  userLocation?: [number, number]; // [lng, lat]
  onReportSuccess?: () => void;
}

export default function QuickReportModal({ 
  onClose, 
  userLocation = [36.8225, -1.2875],
  onReportSuccess 
}: QuickReportModalProps) {
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<'low' | 'medium' | 'high'>('medium');
  const [notes, setNotes] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const maxChars = 200;

  // Map confidence to numeric values
  const confidenceMap = {
    low: 0.3,
    medium: 0.6,
    high: 0.9
  };

  const reportTypes = [
    { id: 'safe', label: 'Safe', icon: CheckCircle, color: 'green' },
    { id: 'crowd', label: 'Crowd', icon: Users, color: 'amber' },
    { id: 'police', label: 'Police', icon: Shield, color: 'blue' },
    { id: 'tear_gas', label: 'Tear Gas', icon: Wind, color: 'red' },
    { id: 'water_cannon', label: 'Water Cannon', icon: Droplets, color: 'red' },
  ];

  useEffect(() => {
    console.log('[QuickReport] User location:', userLocation);
  }, [userLocation]);

  const handleSubmit = async () => {
    if (!selectedType) {
      toast.error('Please select a report type');
      return;
    }

    setSubmitting(true);

    try {
      const [lng, lat] = userLocation;
      
      // Prepare report data - exactly matching backend expectations
      const reportData = {
        type: selectedType,
        lat: lat,  // latitude (y-coordinate)
        lng: lng,  // longitude (x-coordinate)
        confidence: confidenceMap[confidence],
        notes: notes.trim() || undefined,
        timestamp: new Date().toISOString()
      };

      console.log('[QuickReport] Submitting report:', reportData);
      console.log('[QuickReport] Request URL:', 'http://localhost:8000/report');

      const response = await fetch('http://localhost:8000/report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(reportData)
      });

      console.log('[QuickReport] Response status:', response.status);
      
      // Get response text first for debugging
      const responseText = await response.text();
      console.log('[QuickReport] Response text:', responseText);

      if (!response.ok) {
        let errorMessage = `Server returned ${response.status}`;
        try {
          const errorData = JSON.parse(responseText);
          errorMessage = errorData.error || errorMessage;
          console.error('[QuickReport] Error data:', errorData);
        } catch (e) {
          console.error('[QuickReport] Could not parse error response');
        }
        throw new Error(errorMessage);
      }

      // Parse successful response
      let result;
      try {
        result = JSON.parse(responseText);
      } catch (e) {
        console.error('[QuickReport] Could not parse success response');
        throw new Error('Invalid response from server');
      }

      console.log('[QuickReport] Report submitted successfully:', result);

      toast.success('Report received', {
        description: `Your ${selectedType.replace('_', ' ')} report will be visible for ${Math.floor(result.expires_in_seconds / 60)} minutes.`,
      });

      // Trigger map refresh
      if (onReportSuccess) {
        onReportSuccess();
      }

      onClose();

    } catch (error) {
      console.error('[QuickReport] Submission failed:', error);
      
      toast.error('Failed to submit report', {
        description: error instanceof Error ? error.message : 'Please check your connection and try again.'
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <motion.div 
      className="absolute inset-0 z-50 flex items-end justify-center"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      {/* Backdrop */}
      <motion.div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        onClick={onClose}
      />

      {/* Modal Content */}
      <motion.div 
        className="relative w-full bg-white rounded-t-3xl p-6 max-h-[90%] overflow-y-auto shadow-2xl"
        initial={{ y: '100%' }}
        animate={{ y: 0 }}
        exit={{ y: '100%' }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-xl font-bold">Quick Report</h3>
            <p className="text-sm text-neutral-600 mt-1">
              Location: {userLocation[1].toFixed(4)}, {userLocation[0].toFixed(4)}
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="rounded-full"
            disabled={submitting}
            asChild
          >
            <motion.button whileTap={{ scale: 0.9 }}>
              <X className="w-5 h-5" strokeWidth={2} />
            </motion.button>
          </Button>
        </div>

        {/* Report Type Selection */}
        <div className="mb-6">
          <label className="text-sm text-neutral-600 mb-3 block">What are you observing?</label>
          <div className="grid grid-cols-2 gap-3">
            {reportTypes.map((type) => {
              const Icon = type.icon;
              const isSelected = selectedType === type.id;
              
              return (
                <motion.button
                  key={type.id}
                  onClick={() => setSelectedType(type.id)}
                  disabled={submitting}
                  className={`p-4 rounded-xl border-3 transition-all ${
                    isSelected
                      ? type.color === 'green'
                        ? 'border-green-500 bg-green-50'
                        : type.color === 'amber'
                        ? 'border-amber-500 bg-amber-50'
                        : type.color === 'blue'
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-red-500 bg-red-50'
                      : 'border-neutral-200 bg-white hover:border-neutral-300'
                  } ${submitting ? 'opacity-50 cursor-not-allowed' : ''}`}
                  whileHover={!submitting ? { scale: 1.05 } : {}}
                  whileTap={!submitting ? { scale: 0.95 } : {}}
                >
                  <Icon
                    className={`w-6 h-6 mx-auto mb-2 ${
                      isSelected ? 'text-neutral-900' : 'text-neutral-600'
                    }`}
                    strokeWidth={2.5}
                  />
                  <span className={`text-sm block ${isSelected ? 'text-neutral-900 font-medium' : 'text-neutral-700'}`}>
                    {type.label}
                  </span>
                </motion.button>
              );
            })}
          </div>
        </div>

        {/* Confidence Level */}
        <div className="mb-6">
          <label className="text-sm text-neutral-600 mb-3 block">
            How certain are you?
          </label>
          <div className="flex gap-2">
            {(['low', 'medium', 'high'] as const).map((level) => (
              <motion.button
                key={level}
                onClick={() => setConfidence(level)}
                disabled={submitting}
                className={`flex-1 justify-center py-3 rounded-xl border-2 transition-all ${
                  confidence === level
                    ? 'bg-neutral-900 text-white border-neutral-900'
                    : 'bg-white text-neutral-600 border-neutral-300 hover:border-neutral-400'
                } ${submitting ? 'opacity-50 cursor-not-allowed' : ''}`}
                whileTap={!submitting ? { scale: 0.98 } : {}}
              >
                <span className="text-sm font-medium">
                  {level.charAt(0).toUpperCase() + level.slice(1)}
                </span>
                <span className="text-xs block mt-1 opacity-70">
                  {Math.round(confidenceMap[level] * 100)}%
                </span>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Optional Notes */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm text-neutral-600">
              Additional details (optional)
            </label>
            <span className={`text-xs ${
              notes.length > maxChars ? 'text-red-600' : 'text-neutral-500'
            }`}>
              {notes.length}/{maxChars}
            </span>
          </div>
          <textarea
            value={notes}
            onChange={(e) => {
              if (e.target.value.length <= maxChars) {
                setNotes(e.target.value);
              }
            }}
            disabled={submitting}
            placeholder="Add any relevant information..."
            rows={3}
            className="w-full px-4 py-3 border-2 border-neutral-300 rounded-xl resize-none focus:outline-none focus:border-neutral-900 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          />
        </div>

        {/* Submit Button */}
        <Button
          onClick={handleSubmit}
          disabled={submitting || !selectedType}
          className="w-full bg-neutral-900 hover:bg-neutral-800 mb-3 disabled:opacity-50 disabled:cursor-not-allowed"
          size="lg"
          asChild
        >
          <motion.button
            whileHover={!submitting && selectedType ? { scale: 1.02 } : {}}
            whileTap={!submitting && selectedType ? { scale: 0.98 } : {}}
          >
            {submitting ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                Submitting...
              </>
            ) : (
              'Send Anonymously'
            )}
          </motion.button>
        </Button>

        {/* Privacy Notice */}
        <div className="bg-neutral-50 rounded-xl p-3 border-2 border-neutral-200">
          <p className="text-xs text-neutral-700">
            <strong>Privacy:</strong> Reports are anonymous and expire after 10 minutes. 
            Your location is quantized to the nearest street intersection (~{userLocation ? '20m' : 'unknown'}).
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}