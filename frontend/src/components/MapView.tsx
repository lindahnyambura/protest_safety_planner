// frontend/src/components/MapView.tsx
import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import { apiClient, type RiskMapResponse } from '../lib/api';

mapboxgl.accessToken = 'pk.eyJ1IjoibnlhbWJ1cmFsIiwiYSI6ImNtaGV5OGtldDAxNHEyanF2ODZ5eGd0YjYifQ.Tl1_xuqn3wEEzOWh5A9tbA'; // Replace with your token

interface MapViewProps {
  showUserLocation?: boolean;
  showHazards?: boolean;
  showRoute?: boolean;
  routeGeometry?: [number, number][];
  userLocation?: { lat: number; lng: number } | null;
  onLocationUpdate?: (location: { lat: number; lng: number }, name: string) => void;
}

export function MapView({ 
  showUserLocation = true, 
  showHazards = true, 
  showRoute = false, 
  routeGeometry, 
  userLocation,
  onLocationUpdate 
}: MapViewProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [riskData, setRiskData] = useState<RiskMapResponse | null>(null);
  const [isMapLoaded, setIsMapLoaded] = useState(false);

  useEffect(() => {
    if (!mapContainer.current) return;

    // Initialize map
    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/light-v11',
      center: userLocation ? [userLocation.lng, userLocation.lat] : [36.817, -1.283], // Nairobi CBD
      zoom: 15,
    });

    map.current.addControl(new mapboxgl.NavigationControl());

    // Load risk data when map is ready
    map.current.on('load', () => {
      setIsMapLoaded(true);
      if (showHazards) {
        loadRiskData();
      }
      if (showUserLocation && userLocation) {
        addUserLocationMarker();
      }
    });

    // Watch for location updates
    if (showUserLocation && !userLocation) {
      getUserLocation();
    }

    return () => {
      if (map.current) {
        map.current.remove();
      }
    };
  }, []);

  // Update map when user location changes
  useEffect(() => {
    if (map.current && userLocation && isMapLoaded) {
      map.current.flyTo({
        center: [userLocation.lng, userLocation.lat],
        zoom: 15,
        essential: true
      });
      addUserLocationMarker();
    }
  }, [userLocation, isMapLoaded]);

  const getUserLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          if (onLocationUpdate) {
            onLocationUpdate({ lat: latitude, lng: longitude }, "Your Location");
          }
        },
        (error) => {
          console.warn("Geolocation failed:", error);
        }
      );
    }
  };

  const addUserLocationMarker = () => {
    if (!map.current || !userLocation) return;

    // Remove existing user location marker
    if (map.current.getLayer('user-location')) {
      map.current.removeLayer('user-location');
    }
    if (map.current.getSource('user-location')) {
      map.current.removeSource('user-location');
    }

    // Add user location marker
    map.current.addSource('user-location', {
      type: 'geojson',
      data: {
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [userLocation.lng, userLocation.lat]
        },
        properties: {}
      }
    });

    map.current.addLayer({
      id: 'user-location',
      type: 'circle',
      source: 'user-location',
      paint: {
        'circle-radius': 8,
        'circle-color': '#04771B',
        'circle-stroke-color': '#000',
        'circle-stroke-width': 2,
        'circle-opacity': 0.8
      }
    });
  };

  const loadRiskData = async () => {
    try {
      const data = await apiClient.getRiskMap();
      setRiskData(data);
    } catch (error) {
      console.error('Failed to load risk data:', error);
    }
  };

  // Add risk overlay when data loads
  useEffect(() => {
    if (!map.current || !riskData || !showHazards || !isMapLoaded) return;

    const mapInstance = map.current;

    // Add risk data source
    if (!mapInstance.getSource('risk-data')) {
      mapInstance.addSource('risk-data', {
        type: 'geojson',
        data: riskData,
      });

      // Add risk heatmap layer
      mapInstance.addLayer({
        id: 'risk-heatmap',
        type: 'circle',
        source: 'risk-data',
        paint: {
          'circle-radius': [
            'interpolate', ['linear'], ['zoom'],
            14, 6,
            16, 10,
            18, 15
          ],
          'circle-color': [
            'interpolate',
            ['linear'],
            ['get', 'risk'],
            0, '#00ff00',    // Green - low risk
            0.3, '#ffff00',  // Yellow - medium risk
            0.7, '#ff9900',  // Orange - high risk
            1, '#ff0000'     // Red - very high risk
          ],
          'circle-opacity': 0.7,
          'circle-blur': 0.8,
          'circle-stroke-width': 1,
          'circle-stroke-color': '#000'
        },
      });

      // Add risk labels on hover
      mapInstance.addLayer({
        id: 'risk-labels',
        type: 'symbol',
        source: 'risk-data',
        layout: {
          'text-field': ['get', 'intensity'],
          'text-size': 12,
          'text-offset': [0, 1.5],
          'visibility': 'none' // Hidden by default
        },
        paint: {
          'text-color': '#000',
          'text-halo-color': '#fff',
          'text-halo-width': 2
        }
      });
    } else {
      // Update existing source
      (mapInstance.getSource('risk-data') as mapboxgl.GeoJSONSource).setData(riskData);
    }
  }, [riskData, showHazards, isMapLoaded]);

  // Add route layer
  useEffect(() => {
    if (!map.current || !routeGeometry || !showRoute || !isMapLoaded) return;

    const mapInstance = map.current;

    // Create GeoJSON for route
    const routeGeoJSON = {
      type: 'Feature' as const,
      geometry: {
        type: 'LineString' as const,
        coordinates: routeGeometry,
      },
      properties: {},
    };

    // Add or update route source
    if (!mapInstance.getSource('route')) {
      mapInstance.addSource('route', {
        type: 'geojson',
        data: routeGeoJSON,
      });

      mapInstance.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'route',
        layout: {
          'line-join': 'round',
          'line-cap': 'round',
        },
        paint: {
          'line-color': '#04771B',
          'line-width': 4,
          'line-opacity': 0.8,
        },
      });
    } else {
      (mapInstance.getSource('route') as mapboxgl.GeoJSONSource).setData(routeGeoJSON);
    }
  }, [routeGeometry, showRoute, isMapLoaded]);

  return <div ref={mapContainer} className="w-full h-full" />;
}