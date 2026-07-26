import React, { useEffect, useState } from 'react';
import { Activity, BrainCircuit, BarChart3 } from 'lucide-react';
import { api } from '../api/client';
import clsx from 'clsx';

export const Sidebar: React.FC = () => {
  const [isHealthy, setIsHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    api.getHealth()
      .then(() => setIsHealthy(true))
      .catch(() => setIsHealthy(false));
      
    const interval = setInterval(() => {
      api.getHealth()
        .then(() => setIsHealthy(true))
        .catch(() => setIsHealthy(false));
    }, 15000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <aside className="glass-panel sidebar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '3rem' }}>
        <BrainCircuit size={32} color="var(--accent-neutral)" />
        <h2 style={{ fontSize: '1rem', margin: 0, lineHeight: 1.3 }}>Financial Market<br/>Intelligence System</h2>
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: '1rem', flex: 1 }}>
        <a href="#" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--text-primary)', textDecoration: 'none', padding: '0.75rem', background: 'rgba(255,255,255,0.05)', borderRadius: 'var(--radius-sm)' }}>
          <BarChart3 size={20} />
          <span>Dashboard</span>
        </a>
      </nav>

      <div style={{ marginTop: 'auto', padding: '1rem', background: 'rgba(0,0,0,0.2)', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <Activity size={20} className={clsx(isHealthy ? "pred-up" : "pred-down")} />
        <div>
          <div style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>System Status</div>
          <div style={{ fontSize: '0.875rem', fontWeight: 600, color: isHealthy ? 'var(--accent-up)' : 'var(--accent-down)' }}>
            {isHealthy === null ? 'Checking...' : isHealthy ? 'API Online' : 'API Offline'}
          </div>
        </div>
      </div>
    </aside>
  );
};
