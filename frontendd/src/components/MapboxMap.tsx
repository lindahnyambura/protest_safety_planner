import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

// Mapbox token
mapboxgl.accessToken = 'pk.eyJ1IjoibnlhbWJ1cmFsIiwiYSI6ImNtaGV5OGtldDAxNHEyanF2ODZ5eGd0YjYifQ.Tl1_xuqn3wEEzOWh5A9tbA';

interface MapboxMapProps {
  onMapLoad?: (map: mapboxgl.Map) => void;
  userLocation?: [number, number]; // [lng, lat]
  showRiskLayer?: boolean;
}

export default function MapboxMap({ 
  onMapLoad, 
  userLocation,
  showRiskLayer = true 
}: MapboxMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [loaded, setLoaded] = useState(false);

  // Nairobi CBD bounds from your metadata
  const NAIROBI_BOUNDS: [number, number, number, number] = [
    36.81,  // west
    -1.295, // south
    36.835, // east
    -1.28   // north
  ];

  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    // Initialize map
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/nyambural/cmhnsrbkz001o01s4dvswcypq',
      center: [36.8225, -1.2875], // Center of CBD
      zoom: 14,
      maxBounds: NAIROBI_BOUNDS
    });

    map.current.on('load', () => {
      setLoaded(true);
      if (onMapLoad && map.current) {
        onMapLoad(map.current);
      }
    });

    // Add navigation controls
    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    return () => {
      map.current?.remove();
    };
  }, [onMapLoad]);

  // Add user location marker
  useEffect(() => {
    if (!map.current || !loaded || !userLocation) return;

    new mapboxgl.Marker({ color: '#000' })
      .setLngLat(userLocation)
      .addTo(map.current);
  }, [loaded, userLocation]);

  // Load risk heatmap
  useEffect(() => {
    if (!map.current || !loaded || !showRiskLayer) return;

    const loadRiskLayer = async () => {
      try {
        // Fetch bounds first
        const boundsRes = await fetch('http://localhost:8000/riskmap-bounds');
        const boundsData = await boundsRes.json();

        const [west, south, east, north] = boundsData.bounds;

        // Add risk heatmap as image source
        if (map.current) {
          map.current.addSource('risk-heatmap', {
            type: 'image',
            url: 'http://localhost:8000/riskmap-image',
            coordinates: [
              [west, north],  // top-left
              [east, north],  // top-right
              [east, south],  // bottom-right
              [west, south]   // bottom-left
            ]
          });

          map.current.addLayer({
            id: 'risk-heatmap-layer',
            type: 'raster',
            source: 'risk-heatmap',
            paint: {
              'raster-opacity': 0.65,
              'raster-fade-duration': 0
            }
          });
        }
      } catch (error) {
        console.error('Failed to load risk heatmap:', error);
      }
    };

    loadRiskLayer();

    // Cleanup
    return () => {
      if (map.current?.getLayer('risk-heatmap-layer')) {
        map.current.removeLayer('risk-heatmap-layer');
      }
      if (map.current?.getSource('risk-heatmap')) {
        map.current.removeSource('risk-heatmap');
      }
    };
  }, [loaded, showRiskLayer]);

  return (
    <div 
      ref={mapContainer} 
      className="absolute inset-0 w-full h-full"
    />
  );
}