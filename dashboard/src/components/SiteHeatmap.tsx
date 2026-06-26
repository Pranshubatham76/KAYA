import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useStore } from '../store/useStore';

export const SiteHeatmap: React.FC = () => {
  const { events } = useStore();
  const center: [number, number] = [37.7749, -122.4194]; // Default site center

  return (
    <div className="glass-panel p-1 h-full relative overflow-hidden">
      <MapContainer center={center} zoom={15} className="w-full h-full rounded-xl z-0">
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://carto.com/">CartoDB</a>'
        />
        {events.filter(e => e.gps_lat && e.gps_lon).map((event) => (
          <CircleMarker
            key={event.id}
            center={[event.gps_lat, event.gps_lon]}
            radius={8}
            pathOptions={{
              color: event.status === 'CONFIRMED' ? '#10B981' : event.status === 'PENDING' ? '#F59E0B' : '#6B7280',
              fillColor: event.status === 'CONFIRMED' ? '#10B981' : event.status === 'PENDING' ? '#F59E0B' : '#6B7280',
              fillOpacity: 0.7,
            }}
          >
            <Popup className="bg-surface text-white border-none rounded-lg shadow-xl">
              <div className="font-bold text-primary mb-1">{event.yamnet_class}</div>
              <div className="text-xs text-gray-300">Confidence: {(event.anomaly_score * 100).toFixed(1)}%</div>
              <div className="text-xs text-gray-300">Status: {event.status}</div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
      <div className="absolute top-4 left-4 z-[400] bg-background/80 backdrop-blur-md px-4 py-2 rounded-lg border border-white/10 shadow-lg text-sm text-white font-medium">
        Live Site Map
      </div>
    </div>
  );
};
