import React from 'react';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { type HistoryRecord } from '../api/client';

interface Props {
  data: HistoryRecord[];
}

export const StockChart: React.FC<Props> = ({ data }) => {
  // Data is fetched DESC from API, so reverse it for the chart to flow left-to-right (chronological)
  const chartData = [...data].reverse();

  if (!data || data.length === 0) {
    return (
      <div className="glass-panel" style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
        No historical data available.
      </div>
    );
  }

  // Determine trend to color the chart gradient
  const startPrice = chartData[0].close;
  const endPrice = chartData[chartData.length - 1].close;
  const isUp = endPrice >= startPrice;
  const color = isUp ? 'var(--accent-up)' : 'var(--accent-down)';

  return (
    <div className="glass-panel" style={{ height: '400px', width: '100%', display: 'flex', flexDirection: 'column' }}>
      <h3 style={{ marginBottom: '1rem', color: 'var(--text-secondary)', fontSize: '1rem' }}>Price History (120 Days)</h3>
      <div style={{ flex: 1 }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={color} stopOpacity={0.3}/>
                <stop offset="95%" stopColor={color} stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis 
              dataKey="date" 
              tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
              tickLine={false}
              axisLine={false}
              minTickGap={30}
            />
            <YAxis 
              domain={['auto', 'auto']} 
              tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(val) => `$${val.toFixed(0)}`}
              width={60}
            />
            <Tooltip 
              contentStyle={{ background: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)', color: 'var(--text-primary)' }}
              itemStyle={{ color: 'var(--text-primary)', fontWeight: 600 }}
              labelStyle={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}
              formatter={(value: any) => [`$${Number(value).toFixed(2)}`, 'Close']}
            />
            <Area 
              type="monotone" 
              dataKey="close" 
              stroke={color} 
              strokeWidth={2}
              fillOpacity={1} 
              fill="url(#colorClose)" 
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
