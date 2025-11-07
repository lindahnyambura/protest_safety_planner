import { useState } from 'react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { X, CheckCircle, Users, Shield, AlertTriangle, Wind, Info } from 'lucide-react';
import { toast } from 'sonner';
import { motion, AnimatePresence } from 'motion/react';

interface QuickReportModalProps {
  onClose: () => void;
}

export default function QuickReportModal({ onClose }: QuickReportModalProps) {
  const [selectedType, setSelectedType] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<'low' | 'medium' | 'high'>('medium');
  const [notes, setNotes] = useState('');
  const maxChars = 200;

  const reportTypes = [
    { id: 'safe', label: 'Safe', icon: CheckCircle, color: 'green' },
    { id: 'crowd', label: 'Crowd', icon: Users, color: 'amber' },
    { id: 'police', label: 'Police', icon: Shield, color: 'blue' },
    { id: 'tear-gas', label: 'Tear Gas', icon: Wind, color: 'red' },
    { id: 'other', label: 'Other', icon: Info, color: 'neutral' },
  ];

  const handleSubmit = () => {
    if (!selectedType) {
      toast.error('Please select a report type');
      return;
    }

    toast.success('Report received', {
      description: 'Thank you for keeping the community safe.',
    });
    
    onClose();
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
          <h3>Quick Report</h3>
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            className="rounded-full"
            asChild
          >
            <motion.button whileTap={{ scale: 0.9 }}>
              <X className="w-5 h-5" strokeWidth={2} />
            </motion.button>
          </Button>
        </div>

        {/* Report Type Selection */}
        <div className="mb-6">
          <label className="text-sm text-neutral-600 mb-3 block">What are you seeing?</label>
          <div className="grid grid-cols-3 gap-3">
            {reportTypes.map((type) => {
              const Icon = type.icon;
              const isSelected = selectedType === type.id;
              
              return (
                <motion.button
                  key={type.id}
                  onClick={() => setSelectedType(type.id)}
                  className={`p-4 rounded-xl border-3 transition-all ${
                    isSelected
                      ? type.color === 'green'
                        ? 'border-green-500 bg-green-50'
                        : type.color === 'amber'
                        ? 'border-amber-500 bg-amber-50'
                        : type.color === 'blue'
                        ? 'border-blue-500 bg-blue-50'
                        : type.color === 'red'
                        ? 'border-red-500 bg-red-50'
                        : 'border-neutral-500 bg-neutral-50'
                      : 'border-neutral-200 bg-white hover:border-neutral-300'
                  }`}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <Icon
                    className={`w-6 h-6 mx-auto mb-2 ${
                      isSelected ? 'text-neutral-900' : 'text-neutral-600'
                    }`}
                    strokeWidth={2.5}
                  />
                  <span className={`text-sm ${isSelected ? 'text-neutral-900 font-medium' : 'text-neutral-700'}`}>
                    {type.label}
                  </span>
                </motion.button>
              );
            })}
          </div>
        </div>

        {/* Confidence Level */}
        <div className="mb-6">
          <label className="text-sm text-neutral-600 mb-3 block">Confidence</label>
          <div className="flex gap-2">
            {(['low', 'medium', 'high'] as const).map((level) => (
              <motion.button
                key={level}
                onClick={() => setConfidence(level)}
                className={`flex-1 justify-center py-3 rounded-xl border-2 transition-all ${
                  confidence === level
                    ? 'bg-neutral-900 text-white border-neutral-900'
                    : 'bg-white text-neutral-600 border-neutral-300 hover:border-neutral-400'
                }`}
                whileTap={{ scale: 0.98 }}
              >
                <span className="text-sm font-medium">
                  {level.charAt(0).toUpperCase() + level.slice(1)}
                </span>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Optional Notes with Character Counter */}
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
            placeholder="Add any relevant information..."
            rows={3}
            className="w-full px-4 py-3 border-2 border-neutral-300 rounded-xl resize-none focus:outline-none focus:border-neutral-900 transition-colors"
          />
        </div>

        {/* Submit Button */}
        <Button
          onClick={handleSubmit}
          className="w-full bg-neutral-900 hover:bg-neutral-800 mb-3"
          size="lg"
          asChild
        >
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            Send Anonymously
          </motion.button>
        </Button>

        {/* Privacy Notice */}
        <p className="text-xs text-neutral-500 text-center">
          Your report is submitted anonymously and will expire after 10 minutes.
        </p>
      </motion.div>
    </motion.div>
  );
}
