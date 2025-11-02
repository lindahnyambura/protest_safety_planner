import { useState } from "react";
import { motion } from "motion/react";
import { ArrowRight, MapPin, Info } from "lucide-react";
import { ImageWithFallback } from "./figma/ImageWithFallback";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "./ui/carousel";

interface LandingProps {
  onComplete: () => void;
}

// Mock data for current protest - in production this would come from backend
const currentProtest = {
  title: "Saba Saba March",
  date: "July 7, 2025",
  location: "Nairobi CBD",
  description: "Anniversary march of Saba Saba in remembrance of our martyrs. Multiple assembly points across the city.",
  status: "Active",
  posterUrl: "/assets/07-07-25.jpg",
  attendees: "~5,000 expected",
};

// Mock data for past protests
const pastProtests = [
  {
    id: 1,
    title: "The 1 Million People March",
    date: "June 27, 2024",
    summary: "Protests took place in at least 20 counties. Over 500 civilian & police injuries. At least 38 fatalities.",
    status: "Tense",
    statusColor: "#AE1E2A", // red
    thumbnailUrl: "/assets/27-06-24.jpg",
  },
  {
    id: 2,
    title: "June 25th Anniversary March",
    date: "June 25, 2025",
    summary: "Strong turnout in at least 23 counties. Excessive use of force by police. At least 61 arrests made.",
    status: "Tense",
    statusColor: "#AE1E2A", // red
    thumbnailUrl: "/assets/ruto-must-go.jpg",
  },
  {
    id: 3,
    title: "Reject Finance Bill 2024",
    date: "June 18, 2024",
    summary: "Some tensions with authorities; several arrests made. Lost the life of Rex Masai, first casualty of the Gen Z protests.",
    status: "Tense",
    statusColor: "#AE1E2A", // red
    thumbnailUrl: "/assets/reject.jpg",
  },
  {
    id: 4,
    title: "End Femicide March",
    date: "January 27, 2024",
    summary: "Remained peaceful throughout accross the country.",
    status: "Peaceful",
    statusColor: "#04771B", // green
    thumbnailUrl: "/assets/27-01-24.jpg",
  },
];

export function Landing({ onComplete }: LandingProps) {
  const [showPermissionModal, setShowPermissionModal] = useState(false);
  const [showInfo, setShowInfo] = useState(false);

  const handleGetStarted = () => {
    setShowPermissionModal(true);
  };

  const handleAllowLocation = () => {
    onComplete();
  };

  const handleManualLocation = () => {
    onComplete();
  };

  const handleCloseModal = () => {
    setShowPermissionModal(false);
    setShowInfo(false);
  };

  return (
    <div className="min-h-screen bg-[#FDF8F0] overflow-y-auto pb-8">
      {/* Header Section */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 pb-4"
      >
        {/* Small Logo */}
        <div className="flex items-center gap-3 mb-6">
          <div className="border-2 border-black p-2">
            <svg width="32" height="32" viewBox="0 0 50 50" fill="none">
              <path d="M10 40 L20 25 L30 20 L40 10" stroke="black" strokeWidth="3" />
              <circle cx="10" cy="40" r="4" fill="black" stroke="#000" strokeWidth="1.5" />
              <circle cx="40" cy="10" r="4" fill="black" stroke="#000" strokeWidth="1.5" />
              <path d="M30 20 L40 10 L35 18" stroke="black" strokeWidth="2.5" fill="none" />
            </svg>
          </div>
          <div>
            <h1 className="text-2xl" style={{ fontWeight: 900 }}>SAFENAV</h1>
          </div>
        </div>

        {/* Tagline */}
        <div className="border-2 border-black p-4 bg-[#FDF8F0] mb-6">
          <p className="text-sm">
            Real-time protest safety navigation. Community-powered. Anonymous. No tracking.
          </p>
        </div>
      </motion.div>

      {/* Current Protest Hero Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="px-6 mb-6"
      >
        <h2 className="mb-3" style={{ fontWeight: 700 }}>HAPPENING NOW</h2>
        
        <div className="border-3 border-black bg-[#FDF8F0]">
          {/* Poster Image */}
          <div className="border-b-3 border-black relative h-40 overflow-hidden">
            <ImageWithFallback
              src={currentProtest.posterUrl}
              alt={currentProtest.title}
              className="w-full h-full object-cover"
            />
            {/* Status Badge */}
            <div className="absolute top-3 right-3 border-2 border-black bg-[#04771B] px-3 py-1">
              <span className="text-sm" style={{ fontWeight: 700 }}>
                {currentProtest.status}
              </span>
            </div>
          </div>

          {/* Card Content */}
          <div className="p-3">
            <h3 className="mb-1" style={{ fontWeight: 700 }}>{currentProtest.title}</h3>
            <div className="flex items-center gap-2 mb-2 text-sm">
              <span>{currentProtest.date}</span>
              <span>•</span>
              <span>{currentProtest.location}</span>
            </div>
            <p className="text-sm mb-2">{currentProtest.description}</p>
            <div className="border-t-2 border-black pt-2">
              <div className="text-sm">
                <span style={{ fontWeight: 700 }}>Expected attendance:</span> {currentProtest.attendees}
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Past Protests Carousel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mb-6"
      >
        <div className="px-6 mb-2">
          <h2 style={{ fontWeight: 700 }}>PAST PROTESTS</h2>
        </div>

        <div className="px-6">
          <Carousel
            opts={{
              align: "start",
              loop: false,
            }}
            className="w-full"
          >
            <CarouselContent className="-ml-2">
              {pastProtests.map((protest) => (
                <CarouselItem key={protest.id} className="pl-2 basis-[70%]">
                  <div className="border-2 border-black bg-[#FDF8F0] cursor-pointer hover:translate-x-1 hover:translate-y-1 transition-transform">
                    {/* Thumbnail */}
                    <div className="border-b-2 border-black h-20 overflow-hidden">
                      <ImageWithFallback
                        src={protest.thumbnailUrl}
                        alt={protest.title}
                        className="w-full h-full object-cover"
                      />
                    </div>

                    {/* Content */}
                    <div className="p-2">
                      <h4 className="mb-0.5 text-sm" style={{ fontWeight: 700 }}>{protest.title}</h4>
                      <p className="text-xs mb-1">{protest.date}</p>
                      
                      {/* Status Tag */}
                      <div className="inline-block border border-black px-2 py-0.5" style={{ backgroundColor: protest.statusColor }}>
                        <span className="text-xs" style={{ fontWeight: 700 }}>{protest.status}</span>
                      </div>
                    </div>
                  </div>
                </CarouselItem>
              ))}
            </CarouselContent>
          </Carousel>
        </div>
      </motion.div>

      {/* CTA Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="px-6"
      >
        <button
          onClick={handleGetStarted}
          className="w-full border-3 border-black bg-black text-[#FDF8F0] p-4 hover:border-[#04771B] transition-colors flex items-center justify-center gap-2"
          style={{ fontWeight: 700 }}
        >
          <span>GET STARTED</span>
          <ArrowRight size={20} />
        </button>

        <div className="mt-4 text-center">
          <p className="text-sm">
            Join thousands using SafeNav to navigate protests safely
          </p>
        </div>
      </motion.div>

      {/* Location Permission Modal */}
      {showPermissionModal && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          onClick={handleCloseModal}
          className="fixed inset-0 bg-black/60 flex items-center justify-center p-6 z-50"
        >
          <motion.div
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-sm border-3 border-black bg-[#FDF8F0] p-6"
          >
            <div className="flex items-center justify-center mb-6">
              <div className="border-2 border-black p-4 bg-[#04771B]">
                <MapPin size={32} className="text-black" />
              </div>
            </div>
            
            <h2 className="text-xl mb-3 text-center">Location Access</h2>
            <p className="text-sm mb-6 text-center">
              SafeNav needs your location to show hazards and routes near you.
            </p>

            <button
              onClick={handleAllowLocation}
              className="w-full border-3 border-black bg-[rgb(4,119,27)] text-black p-3 mb-3 hover:border-[#000] transition-colors"
            >
              ALLOW LOCATION
            </button>

            <button
              onClick={handleManualLocation}
              className="w-full border-2 border-black bg-[#FDF8F0] text-black p-3 mb-4 hover:bg-[#E8E3D8] transition-colors"
            >
              USE MANUAL LOCATION
            </button>

            {/* Info Button */}
            <button
              onClick={() => setShowInfo(!showInfo)}
              className="w-full border-2 border-black bg-[#FDF8F0] p-2.5 mb-3 flex items-center justify-center gap-2 hover:bg-[#E8E3D8] transition-colors"
            >
              <Info size={16} />
              <span className="text-sm">Why we need your location</span>
            </button>

            {/* Info Expansion */}
            {showInfo && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                className="border-2 border-black p-3 bg-[#E8E3D8]"
              >
                <p className="text-sm">
                  Location data is used to show nearby hazards and compute safe routes. All data stays on your device. No tracking, no servers.
                </p>
              </motion.div>
            )}
          </motion.div>
        </motion.div>
      )}
    </div>
  );
}
