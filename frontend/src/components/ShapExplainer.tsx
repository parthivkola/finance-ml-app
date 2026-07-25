import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';
import { type FeatureImpact } from '../api/client';

interface Props {
  data: FeatureImpact[];
}

export const ShapExplainer: React.FC<Props> = ({ data }) => {
  if (!data || data.length === 0) {
    return null;
  }

  // Format data for Recharts (Bar chart supports negative values natively)
  const chartData = data.map(d => ({
    name: d.feature.replace(/_/g, ' ').toUpperCase(),
    impact: d.impact
  }));

  return (
    <div className="glass-panel" style={{ height: '300px', width: '100%', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ marginBottom: '1rem', color: 'var(--text-secondary)', fontSize: '1rem' }}>
        SHAP Feature Impact
      </h3>
      <div style={{ flex: 1 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 30, left: 30, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={true} vertical={false} />
            <XAxis type="number" tick={{ fill: 'var(--text-muted)' }} axisLine={false} tickLine={false} />
            <YAxis 
              type="category" 
              dataKey="name" 
              tick={{ fill: 'var(--text-primary)', fontSize: 11 }}
              axisLine={false} 
              tickLine={false}
              width={100}
            />
            <Tooltip 
              cursor={{ fill: 'rgba(255,255,255,0.05)' }}
              contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)' }}
              itemStyle={{ fontWeight: 600 }}
              formatter={(value: any) => [value > 0 ? `+${Number(value).toFixed(4)}` : Number(value).toFixed(4), 'Impact']}
            />
            <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.impact > 0 ? 'var(--accent-up)' : 'var(--accent-down)'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center', marginTop: '0.5rem' }}>
        Shows which indicators drove the model's decision (Green = UP pressure, Red = DOWN pressure)
      </div>
    </div>
  );
};
