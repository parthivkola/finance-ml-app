import React, { useEffect, useState } from 'react';
import { api, type ModelMetrics } from '../api/client';

interface Props {
  selectedModel: string;
  onSelect: (modelName: string, overfitStatus?: string | null) => void;
}

export const ModelSelector: React.FC<Props> = ({ selectedModel, onSelect }) => {
  const [models, setModels] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [championName, setChampionName] = useState<string | null>(null);
  const [isFallback, setIsFallback] = useState(false);

  useEffect(() => {
    // Fetch all models + best model in parallel
    Promise.all([api.getModels(), api.getBestModel()])
      .then(([data, best]) => {
        const VALID = /^(xgboost|lightgbm|random_forest|logistic_regression|auto)_(1|3|5)d$/;
        const filtered = data.filter(m => VALID.test(m.model_name));
        const unique = filtered.reduce((acc, curr) => {
          if (!acc[curr.model_name] || acc[curr.model_name].version < curr.version) {
            acc[curr.model_name] = curr;
          }
          return acc;
        }, {} as Record<string, ModelMetrics>);
        const baseModels = Object.values(unique);
        
        setModels(baseModels);

        // Auto-select ensemble champion if nothing is selected yet
        setChampionName(`auto_${best.model_name.split('_').pop()}`);
        setIsFallback(best.is_fallback);
        if (!selectedModel || selectedModel === '') {
          onSelect(`auto_${best.model_name.split('_').pop()}`, 'DYNAMIC');
        }
      })
      .catch(() => {
        // If /models/best fails, fall back to loading models only
        api.getModels().then(data => {
          const VALID = /^(xgboost|lightgbm|random_forest|logistic_regression|auto)_(1|3|5)d$/;
          const filtered = data.filter(m => VALID.test(m.model_name));
          const unique = filtered.reduce((acc, curr) => {
            if (!acc[curr.model_name] || acc[curr.model_name].version < curr.version) {
              acc[curr.model_name] = curr;
            }
            return acc;
          }, {} as Record<string, ModelMetrics>);
          setModels(Object.values(unique));
        });
      })
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loading) {
    return <div style={{ color: 'var(--text-secondary)' }}>Loading models...</div>;
  }

  const selectedData = models.find(m => m.model_name === selectedModel);

  return (
    <div className="glass-panel model-selector-container">
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: '1rem', flexWrap: 'wrap' }}>
        <div>
          <label style={{ display: 'block', fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
            Model
            {championName && (
              <span style={{
                marginLeft: '0.5rem',
                fontSize: '0.7rem',
                background: 'rgba(99,179,237,0.1)',
                color: 'var(--accent-neutral)',
                borderRadius: '4px',
                padding: '1px 6px',
                fontWeight: 500,
                letterSpacing: '0.02em',
                border: '1px solid rgba(99,179,237,0.2)',
              }}>
                ★ champion
              </span>
            )}
          </label>
          <select
            className="input-field"
            value={selectedModel}
            onChange={(e) => {
              const name = e.target.value;
              const m = models.find(m => m.model_name === name);
              onSelect(name, m?.overfit_status ?? null);
            }}
            style={{ width: '260px', textTransform: 'capitalize' }}
          >
            {models.map(m => (
              <option key={m.model_name} value={m.model_name}>
                {m.model_name === championName ? '★ ' : ''}{m.model_name.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
        </div>

        {isFallback && (
          <div style={{
            fontSize: '0.75rem',
            color: '#f6ad55',
            background: 'rgba(246,173,85,0.08)',
            borderRadius: '4px',
            padding: '4px 10px',
            border: '1px solid rgba(246,173,85,0.25)',
          }}>
            ! All models flagged overfit — showing best available
          </div>
        )}
      </div>

      {selectedData && (
        <div className="model-stats-border" style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ACCURACY</div>
            <div style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {selectedData.accuracy ? (selectedData.accuracy * 100).toFixed(1) + '%' : '—'}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>F1 SCORE</div>
            <div style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {selectedData.f1_score?.toFixed(3) || '—'}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ROC-AUC</div>
            <div style={{ fontSize: '1.25rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {selectedData.roc_auc?.toFixed(3) || '—'}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>STATUS</div>
            <div style={{
              fontSize: '1rem',
              fontWeight: 600,
              color: selectedData.overfit_status?.includes('OVERFIT') ? 'var(--accent-down)' : 'var(--accent-up)',
              display: 'flex',
              alignItems: 'center',
              height: '1.25rem',
            }}>
              {selectedData.overfit_status || '✅ OK'}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
