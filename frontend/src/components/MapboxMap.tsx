import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { Users, Shield, Wind, Droplets, CheckCircle, Cross } from 'lucide-react';
import { createRoot } from 'react-dom/client';

mapboxgl.accessToken = 'pk.eyJ1IjoibnlhbWJ1cmFsIiwiYSI6ImNtaGV5OGtldDAxNHEyanF2ODZ5eGd0YjYifQ.Tl1_xuqn3wEEzOWh5A9tbA';

interface ReportMarker {
  id: string;
  type: 'safe' | 'crowd' | 'police' | 'tear_gas' | 'water_cannon';
  lat: number;
  lng: number;
  confidence: number;
  timestamp: number;
  expires_at: number;
  node_id: string;
}

interface MedicalStation {
  id: string;
  name: string;
  lat: number;
  lng: number;
}

interface RouteData {
  geometry_latlng: [number, number][];
  metadata: {
    edge_risks: number[];
    mean_edge_risk: number;
    max_edge_risk: number;
  };
  directions: Array<{
    step: number;
    lat: number;
    lng: number;
    instruction: string;
    street_name: string | null;
  }>;
}

interface MapboxMapProps {
  onMapLoad?: (map: mapboxgl.Map) => void;
  userLocation?: [number, number];
  showRiskLayer?: boolean;
  routeData?: Partial<import('../App').RouteData> | null;
  onWaypointClick?: (step: number) => void;
  reports?: ReportMarker[];
  activeLayers?: string[];
  medicalStations?: MedicalStation[];
}

export default function MapboxMap({ 
  onMapLoad, 
  userLocation,
  showRiskLayer = true,
  routeData = null,
  onWaypointClick,
  reports = [],
  activeLayers = ['risk'],
  medicalStations = []
}: MapboxMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [loaded, setLoaded] = useState(false);
  const markersRef = useRef<mapboxgl.Marker[]>([]);

  const NAIROBI_BOUNDS: [number, number, number, number] = [
    36.80, -1.30, 36.84, -1.27
  ];

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    console.log('[MapboxMap] Initializing map');

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/nyambural/cmhnsrbkz001o01s4dvswcypq',
      center: [36.8225, -1.2875],
      zoom: 14,
      minZoom: 12,
      maxZoom: 18,
      maxBounds: NAIROBI_BOUNDS
    });

    map.current.on('load', () => {
      console.log('[MapboxMap] Map loaded');
      setLoaded(true);
      if (onMapLoad && map.current) {
        onMapLoad(map.current);
      }
    });

    map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

    return () => {
      console.log('[MapboxMap] Cleaning up map');
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, [onMapLoad]);

  // User location marker
  useEffect(() => {
  if (!map.current || !loaded || !userLocation) return;

  const marker = new mapboxgl.Marker({ color: '#000' })
    .setLngLat(userLocation)
    .addTo(map.current);

  // Cleanup properly typed
  return () => {
    marker.remove();
  };
}, [loaded, userLocation]);

  // Risk layer
  useEffect(() => {
    if (!map.current || !loaded) return;

    const RISK_LAYER_ID = 'risk-heatmap-layer';
    const RISK_SOURCE_ID = 'risk-heatmap';

    const loadRiskLayer = async () => {
      if (!map.current || !map.current.getStyle()) return;
      
      try {
        if (map.current.getLayer(RISK_LAYER_ID)) {
          map.current.removeLayer(RISK_LAYER_ID);
        }
        if (map.current.getSource(RISK_SOURCE_ID)) {
          map.current.removeSource(RISK_SOURCE_ID);
        }

        if (!showRiskLayer) return;

        const boundsRes = await fetch('http://localhost:8000/riskmap-bounds');
        const boundsData = await boundsRes.json();
        const [west, south, east, north] = boundsData.bounds;
      
        if (!map.current || !map.current.getStyle()) return;

        map.current.addSource(RISK_SOURCE_ID, {
          type: 'image',
          url: `http://localhost:8000/riskmap-image?t=${Date.now()}`,
          coordinates: [
            [west, north], [east, north], [east, south], [west, south]
          ]
        });

        map.current.addLayer({
          id: RISK_LAYER_ID,
          type: 'raster',
          source: RISK_SOURCE_ID,
          paint: {
            'raster-opacity': 0.65,
            'raster-fade-duration': 0
          }
        });
      
        console.log('[MapboxMap] Risk layer loaded');
      } catch (error) {
        console.error('Risk layer error:', error);
      }
    };

    loadRiskLayer();

    return () => {
      try {
        if (map.current && map.current.getStyle()) {
          if (map.current.getLayer(RISK_LAYER_ID)) {
            map.current.removeLayer(RISK_LAYER_ID);
          }
          if (map.current.getSource(RISK_SOURCE_ID)) {
            map.current.removeSource(RISK_SOURCE_ID);
          }
        }
      } catch (error) {
        // Map destroyed
      }
    };
  }, [loaded, showRiskLayer]);

  // Report markers
  useEffect(() => {
    if (!map.current || !loaded) return;

    // Clear existing markers
    markersRef.current.forEach(marker => marker.remove());
    markersRef.current = [];

    // Helper to create marker icon
    const createMarkerIcon = (type: string, confidence: number) => {
      const el = document.createElement('div');
      el.className = 'report-marker';
      
      const colors = {
        safe: '#16a34a',
        crowd: '#f59e0b',
        police: '#3b82f6',
        tear_gas: '#dc2626',
        water_cannon: '#dc2626'
      };

      const icons = {
        safe: '✓',
        crowd: '👥',
        police: '🛡️',
        tear_gas: '☁️',
        water_cannon: '💧'
      };

      const color = colors[type as keyof typeof colors] || '#737373';
      const icon = icons[type as keyof typeof icons] || '⚠️';
      const size = confidence > 0.7 ? 44 : confidence > 0.4 ? 36 : 28;
      const opacity = 0.5 + (confidence * 0.5); // 0.5 to 1.0

      el.innerHTML = `
        <div style="
          background: ${color};
          border: 3px solid white;
          border-radius: 50%;
          width: ${size}px;
          height: ${size}px;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
          cursor: pointer;
          opacity: ${opacity};
          font-size: ${size * 0.5}px;
        ">
          ${icon}
        </div>
      `;

      return el;
    };

    // Add report markers based on active layers
    reports.forEach(report => {
      if (!activeLayers.includes(report.type)) return;

      const el = createMarkerIcon(report.type, report.confidence);
      
      const timeLeft = Math.max(0, (report.expires_at - Date.now()) / 1000);
      const minutesLeft = Math.floor(timeLeft / 60);

      const popupHTML = `
        <div style="padding: 8px;">
          <strong style="text-transform: capitalize;">${report.type.replace('_', ' ')}</strong><br/>
          <span style="font-size: 12px; color: #666;">
            Confidence: ${Math.round(report.confidence * 100)}%<br/>
            Expires in: ${minutesLeft}m
          </span>
        </div>
      `;

      const marker = new mapboxgl.Marker({ element: el })
        .setLngLat([report.lng, report.lat])
        .setPopup(new mapboxgl.Popup({ offset: 25 }).setHTML(popupHTML));

      if (map.current) {
        marker.addTo(map.current);
        markersRef.current.push(marker);
      }
    });

    console.log(`[MapboxMap] Rendered ${markersRef.current.length} report markers`);

    return () => {
      markersRef.current.forEach(marker => marker.remove());
      markersRef.current = [];
    };
  }, [loaded, reports, activeLayers]);

  // Medical station markers
  useEffect(() => {
    if (!map.current || !loaded || !activeLayers.includes('medical')) return;

    const medicalMarkers: mapboxgl.Marker[] = [];

    medicalStations.forEach(station => {
      const el = document.createElement('div');
      el.className = 'medical-marker';
      el.innerHTML = `
        <div style="
          background: #16a34a;
          border: 3px solid white;
          border-radius: 12px;
          padding: 8px 12px;
          display: flex;
          align-items: center;
          gap: 6px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
          cursor: pointer;
          font-size: 16px;
          color: white;
        ">
          <span></span>
          <span style="font-weight: bold; font-size: 12px;">${station.name}</span>
        </div>
      `;

      const popupHTML = `
        <div style="padding: 8px;">
          <strong>${station.name}</strong><br/>
          <span style="font-size: 12px; color: #666;">Medical assistance available</span>
        </div>
      `;

      const marker = new mapboxgl.Marker({ element: el })
        .setLngLat([station.lng, station.lat])
        .setPopup(new mapboxgl.Popup({ offset: 25 }).setHTML(popupHTML));

      if (map.current) {
        marker.addTo(map.current);
        medicalMarkers.push(marker);
      }
    });

    return () => {
      medicalMarkers.forEach(marker => marker.remove());
    };
  }, [loaded, activeLayers, medicalStations]);

  // Route visualization (existing code)
  useEffect(() => {
    if (
      !map.current ||
      !loaded ||
      !routeData ||
      !routeData.geometry_latlng ||
      !routeData.directions ||
      !routeData.metadata
    ) {
      try {
        if (map.current && map.current.getStyle()) {
          if (map.current.getLayer('route-line')) {
            map.current.removeLayer('route-line');
          }
          if (map.current.getLayer('route-line-outline')) {
            map.current.removeLayer('route-line-outline');
          }
          if (map.current.getSource('route')) {
            map.current.removeSource('route');
          }
        }
        document.querySelectorAll('.route-marker').forEach(el => el.remove());
      } catch (error) {
        // Ignore
      }
      return;
    }

    const route = routeData as RouteData;

    try {
      if (!map.current.getStyle()) return;

      if (map.current.getLayer('route-line-outline')) {
        map.current.removeLayer('route-line-outline');
      }
      if (map.current.getLayer('route-line')) {
        map.current.removeLayer('route-line');
      }
      if (map.current.getSource('route')) {
        map.current.removeSource('route');
      }

      const routeGeoJSON: GeoJSON.Feature<GeoJSON.LineString> = {
        type: 'Feature',
        properties: {
          mean_risk: route.metadata.mean_edge_risk,
          max_risk: route.metadata.max_edge_risk
        },
        geometry: {
          type: 'LineString',
          coordinates: route.geometry_latlng
        }
      };

      map.current.addSource('route', {
        type: 'geojson',
        data: routeGeoJSON
      });

      const maxRisk = route.metadata.max_edge_risk;
      const routeColor = maxRisk < 0.1 ? '#16a34a' : maxRisk < 0.3 ? '#f59e0b' : '#dc2626';
      const routeWidth = maxRisk < 0.1 ? 6 : maxRisk < 0.3 ? 7 : 8;

      map.current.addLayer({
        id: 'route-line-outline',
        type: 'line',
        source: 'route',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': '#000000',
          'line-width': routeWidth + 2,
          'line-opacity': 0.4
        }
      });

      map.current.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'route',
        layout: { 'line-join': 'round', 'line-cap': 'round' },
        paint: {
          'line-color': routeColor,
          'line-width': routeWidth,
          'line-opacity': 0.9
        }
      });

      addWaypointMarkers(route);

      const bounds = new mapboxgl.LngLatBounds();
      route.geometry_latlng.forEach(coord => bounds.extend(coord as [number, number]));
      map.current.fitBounds(bounds, {
        padding: { top: 100, bottom: 100, left: 50, right: 50 },
        duration: 1000
      });

      console.log('[MapboxMap] Route added');
    } catch (error) {
      console.error('Route rendering error:', error);
    }

    return () => {
      try {
        if (map.current && map.current.getStyle()) {
          if (map.current.getLayer('route-line-outline')) {
            map.current.removeLayer('route-line-outline');
          }
          if (map.current.getLayer('route-line')) {
            map.current.removeLayer('route-line');
          }
          if (map.current.getSource('route')) {
            map.current.removeSource('route');
          }
        }
        document.querySelectorAll('.route-marker').forEach(el => el.remove());
      } catch (error) {
        // Map destroyed
      }
    };
  }, [loaded, routeData]);

  const addWaypointMarkers = (route: RouteData) => {
    if (!map.current) return;

    document.querySelectorAll('.route-marker').forEach(el => el.remove());

    const createMarker = (html: string, lngLat: [number, number], popupHTML: string) => {
      if (!map.current) return;
      const el = document.createElement('div');
      el.className = 'route-marker';
      el.innerHTML = html;
      
      new mapboxgl.Marker({ element: el })
        .setLngLat(lngLat)
        .setPopup(new mapboxgl.Popup({ offset: 25 }).setHTML(popupHTML))
        .addTo(map.current);
    };

    const start = route.directions[0];
    if (start) {
      createMarker(
        '<div style="background: #16a34a; border: 3px solid white; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"><span style="color: white; font-weight: bold; font-size: 14px;">S</span></div>',
        [start.lng, start.lat],
        `<strong>Start</strong><br/>${start.street_name || 'Your location'}`
      );
    }

    const end = route.directions[route.directions.length - 1];
    if (end) {
      createMarker(
        '<div style="background: #dc2626; border: 3px solid white; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 8px rgba(0,0,0,0.3);"><span style="color: white; font-weight: bold; font-size: 14px;">E</span></div>',
        [end.lng, end.lat],
        '<strong>Destination</strong>'
      );
    }

    route.directions.slice(1, -1).forEach((dir, idx) => {
      if (dir.instruction.toLowerCase().includes('turn')) {
        const el = document.createElement('div');
        el.className = 'route-marker';
        el.innerHTML = `<div style="background: white; border: 2px solid #000; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"><span style="color: #000; font-weight: bold; font-size: 11px;">${idx + 1}</span></div>`;
        
        el.addEventListener('click', () => {
          if (onWaypointClick) onWaypointClick(dir.step);
        });

        if (map.current) {
          new mapboxgl.Marker({ element: el })
            .setLngLat([dir.lng, dir.lat])
            .setPopup(new mapboxgl.Popup({ offset: 15 }).setHTML(`<strong>Step ${idx + 1}</strong><br/>${dir.instruction}`))
            .addTo(map.current);
        }
      }
    });
  };

  return (
    <div 
      ref={mapContainer} 
      className="absolute inset-0 w-full h-full"
    />
  );
}