import React, { useEffect } from 'react';
import { useStore } from '../store/useStore';
import { AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import axios from 'axios';

export const AlertFeed: React.FC = () => {
  const { events, addEvent, updateEventStatus, activeSite } = useStore();

  useEffect(() => {
    // Setup WebSocket
    const ws = new WebSocket(`ws://127.0.0.1:8000/ws/${activeSite}`);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'NEW_EVENT') {
        addEvent(data.payload);
      }
    };
    return () => ws.close();
  }, [activeSite, addEvent]);

  const handleReview = async (id: string, status: 'CONFIRMED' | 'DISMISSED') => {
    try {
      await axios.put(`http://127.0.0.1:8000/api/v1/events/${id}/review`, {
        status,
        osha_category: 'GENERAL', // Simplify for MVP
        severity: 'MEDIUM',
        review_notes: 'Reviewed via dashboard'
      });
      updateEventStatus(id, status);
    } catch (err) {
      console.error('Failed to review event', err);
    }
  };

  return (
    <div className="glass-panel p-6 h-full flex flex-col">
      <h2 className="text-xl font-bold mb-4 text-white flex items-center gap-2">
        <AlertTriangle className="text-warning" /> Live Alerts
      </h2>
      <div className="flex-1 overflow-y-auto space-y-4 pr-2">
        {events.length === 0 ? (
          <div className="text-gray-400 text-center mt-10">No recent alerts.</div>
        ) : (
          events.map(event => (
            <div key={event.id} className="bg-background/50 border border-white/5 rounded-xl p-4 transition hover:border-primary/50">
              <div className="flex justify-between items-start mb-2">
                <span className="font-semibold text-primary">{event.yamnet_class}</span>
                <span className="text-xs text-gray-400">{new Date(event.timestamp).toLocaleTimeString()}</span>
              </div>
              <div className="text-sm text-gray-300 mb-4">
                Confidence: <span className="text-white font-mono">{(event.anomaly_score * 100).toFixed(1)}%</span>
              </div>
              
              {event.status === 'PENDING' ? (
                <div className="flex gap-2">
                  <button onClick={() => handleReview(event.id, 'CONFIRMED')} className="flex-1 bg-accent/20 text-accent hover:bg-accent hover:text-white px-3 py-2 rounded-lg text-sm font-medium transition flex items-center justify-center gap-2">
                    <CheckCircle size={16} /> Confirm
                  </button>
                  <button onClick={() => handleReview(event.id, 'DISMISSED')} className="flex-1 bg-danger/20 text-danger hover:bg-danger hover:text-white px-3 py-2 rounded-lg text-sm font-medium transition flex items-center justify-center gap-2">
                    <XCircle size={16} /> Dismiss
                  </button>
                </div>
              ) : (
                <div className={`text-sm font-medium flex items-center gap-2 ${event.status === 'CONFIRMED' ? 'text-accent' : 'text-gray-500'}`}>
                  {event.status === 'CONFIRMED' ? <CheckCircle size={16}/> : <XCircle size={16}/>} 
                  {event.status}
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
};
