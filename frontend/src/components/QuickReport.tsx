import { useState } from "react";
import { motion } from "motion/react";
import { CheckCircle, Users, AlertTriangle, Wind, Truck, X } from "lucide-react";
import { toast } from "sonner";

interface QuickReportProps {
  onClose: () => void;
}

type HazardType = "safe" | "crowd" | "police" | "gas" | "water-cannon";
type Confidence = "low" | "medium" | "high";

export function QuickReport({ onClose }: QuickReportProps) {
  const [selectedType, setSelectedType] = useState<HazardType | null>(null);
  const [confidence, setConfidence] = useState<Confidence>("medium");
  const [notes, setNotes] = useState("");

  const hazardTypes = [
    { id: "safe" as HazardType, label: "Safe", icon: CheckCircle, color: "#04771B" },
    { id: "crowd" as HazardType, label: "Crowd", icon: Users, color: "#E8E3D8" },
    { id: "police" as HazardType, label: "Police", icon: AlertTriangle, color: "#19647E" },
    { id: "gas" as HazardType, label: "Tear Gas", icon: Wind, color: "#AE1E2A" },
    { id: "water-cannon" as HazardType, label: "Water Cannon", icon: Truck, color: "#19647E" },
  ];

  const handleSubmit = () => {
    if (!selectedType) {
      toast("Please select a hazard type");
      return;
    }
    toast("Report received", {
      description: "Thank you for keeping others safe",
    });
    setTimeout(() => onClose(), 500);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/60 flex items-end sm:items-center justify-center z-50"
      onClick={onClose}
    >
      <motion.div
        initial={{ y: "100%" }}
        animate={{ y: 0 }}
        exit={{ y: "100%" }}
        transition={{ type: "spring", damping: 25 }}
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md border-3 border-black bg-[#FDF8F0] max-h-[90vh] overflow-y-auto"
      >
        {/* Header */}
        <div className="border-b-3 border-black p-4 flex items-center justify-between bg-[#FDF8F0]">
          <h2 className="text-lg">REPORT HAZARD</h2>
          <button
            onClick={onClose}
            className="border-2 border-black p-2 hover:bg-[#E8E3D8] transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        <div className="p-6">
          {/* Hazard Type Selection */}
          <div className="mb-6">
            <label className="block mb-3 text-sm">SELECT HAZARD TYPE</label>
            <div className="grid grid-cols-3 gap-3">
              {hazardTypes.map((type) => {
                const Icon = type.icon;
                const isSelected = selectedType === type.id;
                return (
                  <motion.button
                    key={type.id}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setSelectedType(type.id)}
                    className="border-2 border-black p-4 flex flex-col items-center gap-2 hover:bg-[#E8E3D8] transition-colors"
                    style={{
                      backgroundColor: isSelected ? type.color : "#FDF8F0",
                      borderWidth: isSelected ? "3px" : "2px",
                    }}
                  >
                    <Icon size={24} />
                    <span className="text-xs">{type.label}</span>
                  </motion.button>
                );
              })}
            </div>
          </div>

          {/* Confidence Level */}
          <div className="mb-6">
            <label className="block mb-3 text-sm">CONFIDENCE LEVEL</label>
            <div className="flex gap-2">
              {(["low", "medium", "high"] as Confidence[]).map((level) => (
                <button
                  key={level}
                  onClick={() => setConfidence(level)}
                  className="flex-1 border-2 border-black p-3 hover:bg-[#E8E3D8] transition-colors uppercase text-sm"
                  style={{
                    backgroundColor: confidence === level ? "#E8E3D8" : "#FDF8F0",
                    borderWidth: confidence === level ? "3px" : "2px",
                  }}
                >
                  {level}
                </button>
              ))}
            </div>
          </div>

          {/* Optional Notes */}
          <div className="mb-6">
            <label className="block mb-3 text-sm">ADDITIONAL NOTES (OPTIONAL)</label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Describe what you're seeing..."
              className="w-full border-2 border-black p-3 bg-[#FDF8F0] min-h-24 focus:border-[#6B7F59] outline-none transition-colors"
              maxLength={200}
            />
            <p className="text-xs mt-1 opacity-60">{notes.length}/200</p>
          </div>

          {/* Submit Button */}
          <motion.button
            whileTap={{ scale: 0.98 }}
            onClick={handleSubmit}
            className="w-full border-3 border-black bg-black text-[#FDF8F0] p-4 hover:bg-[#04771B] hover:text-black transition-colors text-center"
            style={{ fontWeight: 600 }}
          >
            SEND ANONYMOUSLY
          </motion.button>

          {/* Privacy Notice */}
          <p className="text-xs mt-4 text-center opacity-60">
            Reports are ephemeral — automatically deleted after your chosen TTL (default 5 min).
          </p>
        </div>
      </motion.div>
    </motion.div>
  );
}