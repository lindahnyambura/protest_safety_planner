import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import { m } from 'motion/react';

// Mapbox token
mapboxgl.accessToken = 'pk.eyJ1IjoibnlhbWJ1cmFsIiwiYSI6ImNtaGV5OGtldDAxNHEyanF2ODZ5eGd0YjYifQ.Tl1_xuqn3wEEzOWh5A9tbA';

interface RouteData {
  geometry_latlng: [number, number][]; // [lng, lat] pairs
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
  userLocation?: [number, number]; // [lng, lat]
  showRiskLayer?: boolean;
  // Accept RouteData from the App but allow its properties (like geometry_latlng) to be optional
  routeData?: Partial<import('../App').RouteData> | null; // NEW: Route to display (partial to allow undefined fields)
  onWaypointClick?: (step: number) => void; // NEW: Callback for waypoint clicks
}

export default function MapboxMap({ 
  onMapLoad, 
  userLocation,
  showRiskLayer = true,
  routeData = null, // NEW: Route to display
  onWaypointClick // NEW: Callback for waypoint clicks 
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

  // Load risk heatmap - update risk layer effect in MapboxMap
  useEffect(() => {
    if (!map.current || !loaded) return;

    const RISK_LAYER_ID = 'risk-heatmap-layer';
    const RISK_SOURCE_ID = 'risk-heatmap';

    const loadRiskLayer = async () => {
      if (!map.current) return;
      
      try {
        // Remove existing layer/source if present
        if (map.current.getLayer(RISK_LAYER_ID)) {
          map.current.removeLayer(RISK_LAYER_ID);
        }
        if (map.current.getSource(RISK_SOURCE_ID)) {
          map.current.removeSource(RISK_SOURCE_ID);
        }

        // Only add if showRiskLayer is true
        if (!showRiskLayer) return;

        // Fetch bounds
        const boundsRes = await fetch('http://localhost:8000/riskmap-bounds');
        const boundsData = await boundsRes.json();
      
        const [west, south, east, north] = boundsData.bounds;
      
        // Add risk heatmap as image source
        map.current.addSource(RISK_SOURCE_ID, {
          type: 'image',
          url: `http://localhost:8000/riskmap-image?t=${Date.now()}`, // Cache bust
          coordinates: [
            [west, north],  // top-left
            [east, north],  // top-right
            [east, south],  // bottom-right
            [west, south]   // bottom-left
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
        console.error('Failed to load risk heatmap:', error);
      }
    };

    loadRiskLayer();

    // Cleanup
    return () => {
      if (map.current?.getLayer(RISK_LAYER_ID)) {
        map.current.removeLayer(RISK_LAYER_ID);
      }
      if (map.current?.getSource(RISK_SOURCE_ID)) {
        map.current.removeSource(RISK_SOURCE_ID);
      }
    };
  }, [loaded, showRiskLayer]);    // Re-run when showRiskLayer changes

  // Route visualization effect
  useEffect(() => {
    // Ensure routeData contains the fields we need before proceeding
    if (
      !map.current ||
      !loaded ||
      !routeData ||
      !routeData.geometry_latlng ||
      !routeData.directions ||
      !routeData.metadata
    ) {
      // Remove route if no data
      if (map.current && loaded) {
        if (map.current.getLayer('route-line')) {
          map.current.removeLayer('route-line');
        }
        if (map.current.getSource('route')) {
          map.current.removeSource('route');
        }
        // Remove markers
        document.querySelectorAll('.route-marker').forEach(el => el.remove());
      }
      return;
    }

    const addRouteToMap = () => {
      if (
        !map.current ||
        !routeData ||
        !routeData.geometry_latlng ||
        !routeData.directions ||
        !routeData.metadata
      ) return;

      // Cast to full RouteData now that we've validated required fields
      const route = routeData as RouteData;

      // Remove existing route
      if (map.current.getLayer('route-line')) {
        map.current.removeLayer('route-line');
      }
      if (map.current.getSource('route')) {
        map.current.removeSource('route');
      }

      // Add route as GeoJSON
      const routeGeoJSON: GeoJSON.Feature<GeoJSON.LineString> = {
        type: 'Feature',
        properties: {
          mean_risk: route.metadata.mean_edge_risk,
          max_risk: route.metadata.max_edge_risk
        },
        geometry: {
          type: 'LineString',
          coordinates: route.geometry_latlng // Already in [lng, lat] format
        }
      };

      map.current.addSource('route', {
        type: 'geojson',
        data: routeGeoJSON
      });

      // Determine route color based on risk
      const maxRisk = route.metadata.max_edge_risk;
      let routeColor: string;
      let routeWidth: number;

      if (maxRisk < 0.1) {
        routeColor = '#16a34a'; // Green - safe
        routeWidth = 6;
      } else if (maxRisk < 0.3) {
        routeColor = '#f59e0b'; // Amber - caution
        routeWidth = 7;
      } else {
        routeColor = '#dc2626'; // Red - high risk
        routeWidth = 8;
      }

      // Add route line with outline
      map.current.addLayer({
        id: 'route-line-outline',
        type: 'line',
        source: 'route',
        layout: {
          'line-join': 'round',
          'line-cap': 'round'
        },
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
        layout: {
          'line-join': 'round',
          'line-cap': 'round'
        },
        paint: {
          'line-color': routeColor,
          'line-width': routeWidth,
          'line-opacity': 0.9
        }
      });

      // Add waypoint markers
      addWaypointMarkers(route);

      // Fit map to route bounds
      const bounds = new mapboxgl.LngLatBounds();
      route.geometry_latlng.forEach(coord => {
        bounds.extend(coord as [number, number]);
      });
      map.current.fitBounds(bounds, {
        padding: { top: 100, bottom: 100, left: 50, right: 50 },
        duration: 1000
      });

      console.log('[MapboxMap] Route added to map');
    };

    addRouteToMap();

    // Cleanup
    return () => {
      if (map.current) {
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
    };
  }, [loaded, routeData]);

  // NEW: Function to add waypoint markers
  const addWaypointMarkers = (route: RouteData) => {
    if (!map.current) return;

    // Remove old markers
    document.querySelectorAll('.route-marker').forEach(el => el.remove());

    // Add start marker
    const startDirection = route.directions[0];
    if (startDirection && map.current) {
      const startEl = document.createElement('div');
      startEl.className = 'route-marker';
      startEl.innerHTML = `
        <div style="
          background: #16a34a;
          border: 3px solid white;
          border-radius: 50%;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
          cursor: pointer;
        ">
          <span style="color: white; font-weight: bold; font-size: 14px;">S</span>
        </div>
      `;

      new mapboxgl.Marker({ element: startEl })
        .setLngLat([startDirection.lng, startDirection.lat])
        .setPopup(
          new mapboxgl.Popup({ offset: 25 })
            .setHTML(`<strong>Start</strong><br/>${startDirection.street_name || 'Your location'}`)
        )
        .addTo(map.current);
    }

    // Add end marker
    const endDirection = route.directions[route.directions.length - 1];
    if (endDirection && map.current) {
      const endEl = document.createElement('div');
      endEl.className = 'route-marker';
      endEl.innerHTML = `
        <div style="
          background: #dc2626;
          border: 3px solid white;
          border-radius: 50%;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
          cursor: pointer;
        ">
          <span style="color: white; font-weight: bold; font-size: 14px;">E</span>
        </div>
      `;

      new mapboxgl.Marker({ element: endEl })
        .setLngLat([endDirection.lng, endDirection.lat])
        .setPopup(
          new mapboxgl.Popup({ offset: 25 })
            .setHTML(`<strong>Destination</strong>`)
        )
        .addTo(map.current);
    }

    // Add intermediate turn markers (only for actual turns)
    route.directions.slice(1, -1).forEach((dir, idx) => {
      if (dir.instruction.toLowerCase().includes('turn') && map.current) {
        const turnEl = document.createElement('div');
        turnEl.className = 'route-marker';
        turnEl.innerHTML = `
          <div style="
            background: white;
            border: 2px solid #000;
            border-radius: 50%;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            cursor: pointer;
          ">
            <span style="color: #000; font-weight: bold; font-size: 11px;">${idx + 1}</span>
          </div>
        `;

        turnEl.addEventListener('click', () => {
          if (onWaypointClick) {
            onWaypointClick(dir.step);
          }
        });

        new mapboxgl.Marker({ element: turnEl })
          .setLngLat([dir.lng, dir.lat])
          .setPopup(
            new mapboxgl.Popup({ offset: 15 })
              .setHTML(`<strong>Step ${idx + 1}</strong><br/>${dir.instruction}`)
          )
          .addTo(map.current);
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