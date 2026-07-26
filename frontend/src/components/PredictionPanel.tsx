import React, { useState } from 'react';
import { Search, Zap } from 'lucide-react';
import { api, type PredictResponse } from '../api/client';
import clsx from 'clsx';

interface Props {
  selectedModel: string;
  onPredictionComplete: (symbol: string, prediction: PredictResponse | null) => void;
}

export const PredictionPanel: React.FC<Props> = ({ selectedModel, onPredictionComplete }) => {
  const [symbol, setSymbol] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symbol) return;

    setLoading(true);
    setError(null);
    setResult(null);
    
    // Clear out previous data in parent
    onPredictionComplete(symbol.toUpperCase(), null);

    try {
      const data = await api.predict(symbol, selectedModel);
      setResult(data);
      onPredictionComplete(symbol.toUpperCase(), data);
    } catch (err: any) {
      console.error(err);
      setError(err.response?.data?.detail || err.message || "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Market Intelligence Engine</h2>
        <p>Enter a stock symbol to generate an AI-driven movement prediction.</p>
      </div>

      <form onSubmit={handleAnalyze} style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', flex: 1, minWidth: '200px' }}>
          <Search size={20} color="var(--text-muted)" style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)' }} />
          <input 
            type="text" 
            className="input-field" 
            placeholder="e.g., AAPL, MSFT, TSLA" 
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            style={{ width: '100%', paddingLeft: '3rem', fontSize: '1.125rem' }}
          />
        </div>
        <button type="submit" className="btn-primary" disabled={loading || !symbol} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {loading ? (
            <span style={{ animation: 'spin 1s linear infinite' }}>⏳</span>
          ) : (
            <Zap size={20} />
          )}
          <span>{loading ? 'Analyzing...' : 'Analyze'}</span>
        </button>
      </form>

      {error && (
        <div style={{ padding: '1rem', background: 'rgba(239, 68, 68, 0.1)', color: 'var(--accent-down)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{ 
          marginTop: '1rem',
          padding: '2rem', 
          background: 'rgba(0,0,0,0.2)', 
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1rem'
        }}>
          <div style={{ fontSize: '1rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '2px' }}>
            Next-Day Prediction
          </div>
          <div className={clsx("pred-" + result.prediction.toLowerCase())} style={{ fontSize: '4rem', fontWeight: 700, lineHeight: 1 }}>
            {result.prediction}
          </div>
          <div style={{ fontSize: '1.125rem', color: 'var(--text-primary)' }}>
            Confidence: <span style={{ fontWeight: 600 }}>{(result.confidence * 100).toFixed(1)}%</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '1rem', textAlign: 'center' }}>
            {result.disclaimer}
          </div>
          
          {result.indicators && (
            <div style={{ 
              display: 'flex', 
              flexWrap: 'wrap',
              gap: '1.5rem', 
              marginTop: '1.5rem', 
              padding: '1rem', 
              background: 'rgba(255,255,255,0.03)', 
              borderRadius: 'var(--radius-sm)',
              width: '100%',
              justifyContent: 'space-around',
              border: '1px solid rgba(255,255,255,0.05)'
            }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>SMA (20)</div>
                <div style={{ fontSize: '1.125rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                  {result.indicators.sma_20 !== null ? result.indicators.sma_20.toFixed(2) : 'N/A'}
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>SMA (50)</div>
                <div style={{ fontSize: '1.125rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                  {result.indicators.sma_50 !== null ? result.indicators.sma_50.toFixed(2) : 'N/A'}
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>RSI (14)</div>
                <div style={{ fontSize: '1.125rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                  {result.indicators.rsi !== null ? result.indicators.rsi.toFixed(2) : 'N/A'}
                </div>
              </div>
              <div style={{ textAlign: 'center' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>MACD</div>
                <div style={{ fontSize: '1.125rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                  {result.indicators.macd !== null ? result.indicators.macd.toFixed(2) : 'N/A'}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
