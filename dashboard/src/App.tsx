import React from 'react';
import { AlertFeed } from './components/AlertFeed';
import { SiteHeatmap } from './components/SiteHeatmap';
import { Analytics } from './components/Analytics';
import { AdminTraining } from './components/AdminTraining';
import { Shield } from 'lucide-react';

function App() {
  return (
    <div className="min-h-screen bg-background text-white flex flex-col p-4 md:p-8">
      {/* Header */}
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/20 flex items-center justify-center border border-primary/50 shadow-[0_0_15px_rgba(59,130,246,0.3)]">
            <Shield className="text-primary" size={24} />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
              SentinelSite
            </h1>
            <p className="text-sm text-primary font-medium">Supervisor Dashboard</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="px-4 py-2 rounded-full bg-surface border border-white/10 text-sm font-medium flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-accent animate-pulse"></div>
            System Online
          </div>
        </div>
      </header>

      {/* Main Grid Layout */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Alerts (4 cols) */}
        <div className="lg:col-span-4 h-[600px] lg:h-auto">
          <AlertFeed />
        </div>

        {/* Middle Column: Map & Analytics (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-6 h-[800px] lg:h-auto">
          <div className="flex-[2] min-h-[400px]">
            <SiteHeatmap />
          </div>
          <div className="flex-[1] min-h-[250px]">
            <Analytics />
          </div>
        </div>

        {/* Right Column: Admin & Training (3 cols) */}
        <div className="lg:col-span-3 h-[400px] lg:h-auto">
          <AdminTraining />
        </div>

      </div>
    </div>
  );
}

export default App;
