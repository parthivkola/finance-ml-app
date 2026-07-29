import React, { useState } from 'react';
import { Search, Zap } from 'lucide-react';
import { api, type PredictResponse, type ExplainResponse } from '../api/client';
import clsx from 'clsx';

interface Props {
  selectedModel: string;
  selectedModelOverfitStatus?: string | null;
  onPredictionComplete: (symbol: string, prediction: PredictResponse | null) => void;
}

export const PredictionPanel: React.FC<Props> = ({
  selectedModel,
  selectedModelOverfitStatus,
  onPredictionComplete,
}) => {
  const [symbol, setSymbol] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);

  // LLM explanation state
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symbol) return;

    setLoading(true);
    setError(null);
    setResult(null);
    setExplanation(null);
    onPredictionComplete(symbol.toUpperCase(), null);

    try {
      const data = await api.predict(symbol, selectedModel);
      setResult(data);
      onPredictionComplete(symbol.toUpperCase(), data);

      // Auto-trigger LLM explanation in the background
      setExplainLoading(true);
      api.explain({
        symbol: symbol.toUpperCase(),
        model_name: data.model_name,
        prediction: data.prediction,
        confidence: data.confidence,
        overfit_status: selectedModelOverfitStatus ?? null,
        explanation: data.explanation ?? [],
        indicators: data.indicators ?? null,
      })
        .then(exp => setExplanation(exp))
        .catch(() => setExplanation(null))   // silent — SHAP chart still shows
        .finally(() => setExplainLoading(false));

    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const horizon =
    selectedModel.includes('3d') ? '3-Day' :
    selectedModel.includes('5d') ? '5-Day' : 'Next-Day';

  return (
    <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>Market Analysis</h2>
        <p>Enter a ticker symbol to run a multi-factor movement analysis.</p>
      </div>

      <form onSubmit={handleAnalyze} style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
        <div style={{ position: 'relative', flex: 1, minWidth: '200px' }}>
          <Search size={18} color="var(--text-muted)" style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)' }} />
          <input
            type="text"
            className="input-field"
            placeholder="e.g., AAPL, MSFT, TSLA"
            value={symbol}
            onChange={e => setSymbol(e.target.value.toUpperCase())}
            style={{ width: '100%', paddingLeft: '3rem', fontSize: '1.125rem' }}
          />
        </div>
        <button
          type="submit"
          className="btn-primary"
          disabled={loading || !symbol}
          style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
        >
          {loading ? <span className="btn-spinner" /> : <Zap size={18} />}
          <span>{loading ? 'Running...' : 'Analyze'}</span>
        </button>
      </form>

      {error && (
        <div style={{ padding: '1rem', background: 'rgba(239,68,68,0.1)', color: 'var(--accent-down)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(239,68,68,0.2)' }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{
          marginTop: '0.5rem',
          padding: '2rem',
          background: 'rgba(0,0,0,0.2)',
          borderRadius: 'var(--radius-md)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '1rem',
        }}>
          {/* Forecast label */}
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '2px' }}>
            {horizon} Forecast
          </div>

          {/* Prediction */}
          <div
            className={clsx('pred-' + result.prediction.toLowerCase())}
            style={{ fontSize: '3.5rem', fontWeight: 700, lineHeight: 1, display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          >
            <span style={{ fontSize: '2.5rem' }}>{result.prediction === 'UP' ? '▲' : '▼'}</span>
            {result.prediction}
          </div>

          {/* Confidence */}
          <div style={{ fontSize: '1rem', color: 'var(--text-secondary)' }}>
            Confidence&ensp;<span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{(result.confidence * 100).toFixed(1)}%</span>
          </div>

          {/* Technical indicators */}
          {result.indicators && (
            <div style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '1.5rem',
              marginTop: '0.5rem',
              padding: '1rem',
              background: 'rgba(255,255,255,0.03)',
              borderRadius: 'var(--radius-sm)',
              width: '100%',
              justifyContent: 'space-around',
              border: '1px solid rgba(255,255,255,0.05)',
            }}>
              {[
                { label: 'SMA (20)', val: result.indicators.sma_20 },
                { label: 'SMA (50)', val: result.indicators.sma_50 },
                { label: 'RSI (14)', val: result.indicators.rsi },
                { label: 'MACD',    val: result.indicators.macd },
              ].map(({ label, val }) => (
                <div key={label} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>{label}</div>
                  <div style={{ fontSize: '1.1rem', color: 'var(--text-primary)', fontWeight: 600 }}>
                    {val !== null && val !== undefined ? val.toFixed(2) : '—'}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* LLM analyst brief — auto-loads after prediction */}
          {(explainLoading || explanation) && (
            <div style={{
              marginTop: '0.75rem',
              padding: '1.25rem 1.5rem',
              background: 'rgba(59,130,246,0.05)',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid rgba(59,130,246,0.12)',
              width: '100%',
            }}>
              <div style={{
                fontSize: '0.7rem',
                color: 'var(--text-muted)',
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
                marginBottom: '0.6rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
              }}>
                <span>Analyst Brief</span>
                {explanation?.provider && (
                  <span style={{
                    fontSize: '0.65rem',
                    background: 'rgba(255,255,255,0.06)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    borderRadius: '3px',
                    padding: '1px 5px',
                    color: 'var(--text-muted)',
                  }}>
                    via {explanation.provider}
                  </span>
                )}
              </div>

              {explainLoading ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
                  <span className="btn-spinner" style={{ borderColor: 'rgba(255,255,255,0.15)', borderTopColor: 'var(--text-muted)' }} />
                  Generating analysis...
                </div>
              ) : (
                <p style={{
                  fontSize: '0.95rem',
                  lineHeight: 1.7,
                  color: 'var(--text-primary)',
                  margin: 0,
                  whiteSpace: 'pre-wrap',
                }}>
                  {explanation?.summary}
                </p>
              )}
            </div>
          )}

          {/* Disclaimer */}
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', textAlign: 'center', maxWidth: '380px' }}>
            {result.disclaimer}
          </div>
        </div>
      )}
    </div>
  );
};
