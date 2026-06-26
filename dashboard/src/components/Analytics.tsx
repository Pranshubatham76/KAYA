import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { useStore } from '../store/useStore';
import { Activity } from 'lucide-react';

export const Analytics: React.FC = () => {
  const { events } = useStore();

  const data = React.useMemo(() => {
    const counts: Record<string, number> = {};
    events.forEach(e => {
      counts[e.yamnet_class] = (counts[e.yamnet_class] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 5);
  }, [events]);

  return (
    <div className="glass-panel p-6 h-full flex flex-col">
      <h2 className="text-xl font-bold mb-4 text-white flex items-center gap-2">
        <Activity className="text-primary" /> Top Incident Types
      </h2>
      <div className="flex-1 w-full min-h-[200px]">
        {data.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-400">Not enough data</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" horizontal={false} />
              <XAxis type="number" stroke="#ffffff50" />
              <YAxis dataKey="name" type="category" stroke="#ffffff90" width={120} tick={{ fill: '#ffffff90', fontSize: 12 }} />
              <Tooltip 
                cursor={{ fill: '#ffffff10' }} 
                contentStyle={{ backgroundColor: '#1A2235', border: '1px solid #ffffff20', borderRadius: '8px', color: '#fff' }} 
              />
              <Bar dataKey="count" fill="#3B82F6" radius={[0, 4, 4, 0]} barSize={20} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
};
